mutable struct iMPO <: AbstractProjMPO
    niter::Int
    lpos::Int
    rpos::Int
    nsite::Int
    nunitcell::Int
    H0::MPO #initial MPO for finite system
    H::MPO #this should be the bulk MPO
    L0::ITensor
    R0::ITensor
    LR::Vector{ITensor}
end
function iMPO(H_ini, H_bulk, Nuc::Int; site=2)
    mpo = iMPO(0, 0, Nuc + 1, site, Nuc, H_ini, H_bulk,
        ITensor(1.0), ITensor(1.0), Vector{ITensor}(undef, Nuc))
    return mpo
end

function ITensorMPS.lproj(P::iMPO)
    (P.lpos <= 0) && return P.L0
    return P.LR[P.lpos]
end
function ITensorMPS.rproj(P::iMPO)
    (P.rpos >= length(P) + 1) && return P.R0
    return P.LR[P.rpos]
end
function update_env!(poi_swap::Int, P::iMPO, H0, psi::MPS)
    #construct env MPO
    #for initial calculations
    N = length(psi)
    # k = Int(N / 2)
    k = poi_swap
    P.lpos = 0
    P.rpos = N + 1
    ll = 0
    L = P.niter > 0 ? P.L0 : OneITensor()
    while ll < k
        L = ((L * psi[ll+1]) * H0[ll+1]) * dag(prime(psi[ll+1]))
        ll += 1
    end
    P.L0 = L
    #for the right 
    rl = N + 1
    R = P.niter > 0 ? P.R0 : OneITensor()
    while rl > k + 1
        R = ((R * psi[rl-1]) * H0[rl-1]) * dag(prime(psi[rl-1]))
        # @show rl-1
        rl -= 1
    end
    P.R0 = R
    return nothing
end
function initializeHam(h::MPO, Nuc, sites_uc; site_start=1, ham_uc=1)
    #find a way to update the hamiltonian indices
    #first find the far left and far right indices
    #construct the bulk mpo by extracting the finite size MPO
    h0 = MPO(Nuc)
    #the bulk mpo should be uniform
    N = length(h)
    site_end = ham_uc + site_start - 1
    uniform_mpo = MPO(ham_uc)
    for i in 1:ham_uc
        uniform_mpo[i] = copy(h[site_start+i-1])
    end
    len_uni_mpo = length(uniform_mpo)
    link = commonind(uniform_mpo[1], h[site_start-1])
    rink = commonind(uniform_mpo[len_uni_mpo], h[site_end+1])
    #
    id = 1
    for i in 1:Nuc
        h0[i] = copy(uniform_mpo[id])
        id += 1
        if id == ham_uc + 1
            #modify rink
            # if i != Nuc
            nlink = settags(new_ind(link), "link,l=$i")
            replaceind!(h0[i], rink, dag(nlink))
            replaceind!(uniform_mpo[1], link, nlink)
            link = nlink
            # end
            id = 1
        end
    end
    for i in 1:Nuc
        site_ids = isiteinds(h0[i:i])[1]
        replaceind!(h0[i], dag(site_ids), dag(sites_uc[i]))
        replaceind!(h0[i], prime(site_ids), prime(sites_uc[i]))
    end
    return h0
end
function swap_MPO!(sites_odd, sites_new, poi_swap, psi::MPS, P::iMPO)
    #swap the MPO 
    #such that before ...H5-a-H1,H2-c-H3,H4,H5-b-H1...
    #is swap to -c-H3,H4,H5-b, a-H1,H2-c
    H_bulk = P.H
    N = length(H_bulk)
    sites_odd = copy(sites_odd)
    dag_site = dag.(sites_new)
    prim_site = prime.(sites_new)

    lind0, rind0 = mpo_env_linkinds(psi, P)
    #left most indices
    mpo_link_lind = setdiff(inds(H_bulk[1]), [dag(sites_odd[1]), prime(sites_odd[1]), commonind(H_bulk[1],
        H_bulk[2])])[1]
    #right most indices
    mpo_link_rind = setdiff(inds(H_bulk[N]), [dag(sites_odd[N]), prime(sites_odd[N]), commonind(H_bulk[N],
        H_bulk[N-1])])[1]
    #swap
    link_id = nothing
    if N == 2
        link_id = commonind(H_bulk[1], H_bulk[2])
    end
    P.H = circshift(P.H, -poi_swap)
    #now modify the link indices make sure mpo can be connected
    replaceind!(P.H[N-poi_swap], mpo_link_rind, dag(mpo_link_lind))
    sites_odd = circshift!(sites_odd, -poi_swap)
    for i in 1:N
        replaceind!(P.H[i], dag(sites_odd[i]), dag_site[i])
        replaceind!(P.H[i], prime(sites_odd[i]), prim_site[i])
    end
    if N == 2
        new_lind = new_ind(link_id)
        new_lind = settags(new_lind, tags(link_id))
        replaceind!(P.H[N], link_id, new_lind)
        replaceind!(P.R0, rind0, dag(new_lind))
    end
    # new_lind = new_ind(mpo_link_rind)
    # new_lind = settags(new_lind, tags(mpo_link_rind))
    # replaceind!(H_bulk[N], mpo_link_rind, new_lind)
    # lind0, rind0 = mpo_env_linkinds(psi, P)
    # replaceind!(P.R0, rind0, dag(new_lind))
    #replace site inds
    P.lpos = 0
    P.rpos = N + 1
    P.niter += 1
    return nothing
end
function initializeMPOLeft!(psi_left, H_start::MPO, mpo::iMPO)
    #H_start is the initial finite size MPO
    #used to initialize the enviroment to minic iDMGR growth steps
    # psi_left, lambda = left_canonical(psi)
    # psi_right, R = right_canonical(psi, lambda0)
    # #left enviroment
    site_psi = isiteinds(psi_left)
    site_mpo = isiteinds(H_start)
    Nsite = length(mpo)
    N_start_size = Int(length(H_start) / 2)
    m = Int(N_start_size / Nsite)
    #for simplicity assuming
    #the N_start_size/N should be an integer
    #otherwise therewill be an error
    #matchining the mpo site indices wit mps site indices
    #first for boundary part
    #first prepare the left initial tensor
    lind = setdiff(uniqueinds(psi_left[1], psi_left[2]), site_psi)[1]
    rind = setdiff(uniqueinds(psi_left[end], psi_left[end-1]), site_psi)[1]
    L = delta(prime(lind), dag(lind))
    count = 1
    for i in 1:N_start_size
        if count > Nsite
            count = 1
        end
        psi_sind = site_psi[count]
        mpo_sind = site_mpo[i]
        A = psi_left[count]
        L = ((L * A) * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                        delta(dag(prime(mpo_sind)), prime(psi_sind)))) * dag(prime(A))
        if count == Nsite
            #place rind to lind
            replaceind!(L, rind, dag(lind))
            replaceind!(L, dag(prime(rind)), prime(lind))
        end
        #constract H with counts psi
        count += 1
    end
    #normalize the L using effective system size to avoid the problem of jordan-block
    # L /= sqrt(2N_start_size)
    #now contract the bulk MPO
    #mpo far left and right link indices
    L_linkind = setdiff(inds(L), [dag(lind), prime(lind)])[1]
    site_mpo = isiteinds(mpo.H)
    pmpo_sind = prime.(site_mpo)
    mpo_lind = setdiff(uniqueinds(mpo.H[1], mpo.H[2]), [dag(site_mpo[1]), pmpo_sind[1]])[1]
    mpo_rind = setdiff(uniqueinds(mpo.H[end], mpo.H[end-1]), [dag(site_mpo[end]), pmpo_sind[end]])[1]
    #match mpo link ind
    replaceind!(L, L_linkind, dag(mpo_lind))
    #align the bulk mpo site indes
    for i in 1:Nsite
        replaceind!(mpo.H[i], dag(site_mpo[i]), dag(site_psi[i]))
        replaceind!(mpo.H[i], prime(site_mpo[i]), prime(site_psi[i]))
    end
    mpo.L0 = L
    return nothing
end
function initializeMPORight!(psi_left, H_start::MPO, mpo::iMPO)
    #H_start is the initial finite size MPO
    #used to initialize the enviroment to minic iDMGR growth steps
    # psi_left, lambda = right_canonical(psi)
    # #left enviroment
    site_psi = isiteinds(psi_left)
    site_mpo = isiteinds(H_start)
    Nsite = length(mpo)
    N_h0 = length(H_start)
    N_start_size = Int(length(H_start) / 2)
    m = Int(N_start_size / Nsite)
    #for simplicity assuming
    #the N_start_size/N should be an integer
    #otherwise therewill be an error
    #matchining the mpo site indices wit mps site indices
    #first for boundary part
    #first prepare the left initial tensor
    lind = setdiff(uniqueinds(psi_left[1], psi_left[2]), site_psi)[1]
    rind = setdiff(uniqueinds(psi_left[end], psi_left[end-1]), site_psi)[1]
    L = delta(prime(rind), dag(rind))
    count = Nsite
    for i in N_h0:-1:N_start_size+1
        if count == 0
            count = Nsite
        end
        psi_sind = site_psi[count]
        mpo_sind = site_mpo[i]
        A = psi_left[count]
        L = ((L * A) * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                        delta(dag(prime(mpo_sind)), prime(psi_sind)))) * dag(prime(A))
        if count == 1
            #place rind to lind
            replaceind!(L, lind, dag(rind))
            replaceind!(L, dag(prime(lind)), prime(rind))
        end
        #constract H with counts psi
        count -= 1
    end
    #normalize the L using effective system size to avoid the problem of jordan-block
    # L /= sqrt(2N_start_size)
    #now contract the bulk MPO
    #mpo far left and right link indices
    L_linkind = setdiff(inds(L), [dag(rind), prime(rind)])[1]
    site_mpo = isiteinds(mpo.H)
    pmpo_sind = prime.(site_mpo)
    mpo_lind = setdiff(uniqueinds(mpo.H[1], mpo.H[2]), [dag(site_mpo[1]), pmpo_sind[1]])[1]
    mpo_rind = setdiff(uniqueinds(mpo.H[end], mpo.H[end-1]), [dag(site_mpo[end]), pmpo_sind[end]])[1]
    #match mpo link ind
    replaceind!(L, L_linkind, dag(mpo_rind))
    mpo.R0 = L
    return nothing
end
function energyMPOSubtraction!(H::MPO, en_density)
    #substract the energy density in current mpo hamilonian
    #for ITensor, the local term is in W[end, 1, :, :]
    Nsite = length(H)
    site_mpo = isiteinds(H)
    pmpo_sind = prime.(site_mpo)
    for i in 1:Nsite
        lind = i != 1 ? commonind(H[i], H[i-1]) : nothing
        rind = i != Nsite ? commonind(H[i], H[i+1]) : nothing
        sind = site_mpo[i]
        dsind = dag(sind)
        psind = prime(sind)
        left_dim = isnothing(lind) ? 1 : dim(lind)
        for s in 1:dim(sind) #site indices
            if i == 1
                H[i][rind=>1, dsind=>s, psind=>s] += -en_density
            elseif i == Nsite
                H[i][lind=>left_dim, dsind=>s, psind=>s] += -en_density
            else
                H[i][lind=>left_dim, rind=>1, dsind=>s, psind=>s] += -en_density
            end
        end
    end
    return nothing
end

function energyMPOSubtraction!(mpo::iMPO, en_density)
    #substract the energy density in current mpo hamilonian
    #for ITensor, the local term is in W[end, 1, :, :]
    Nsite = length(mpo)
    site_mpo = isiteinds(mpo.H)
    pmpo_sind = prime.(site_mpo)
    mpo_lind = setdiff(uniqueinds(mpo.H[1], mpo.H[2]), [dag(site_mpo[1]), pmpo_sind[1]])[1]
    mpo_rind = setdiff(uniqueinds(mpo.H[end], mpo.H[end-1]), [dag(site_mpo[end]), pmpo_sind[end]])[1]
    for i in 1:Nsite
        lind = i != 1 ? commonind(mpo.H[i], mpo.H[i-1]) : mpo_lind
        rind = i != Nsite ? commonind(mpo.H[i], mpo.H[i+1]) : mpo_rind
        sind = site_mpo[i]
        dsind = dag(sind)
        psind = prime(sind)
        left_dim = dim(lind)
        for s in 1:dim(sind) #site indices
            mpo.H[i][lind=>left_dim, rind=>1, dsind=>s, psind=>s] += -en_density
            # @show mpo.H[i][lind=>left_dim, rind=>1, dsind=>s,psind=>s ]        
        end
    end
    return nothing
end
function initializeMPOLeftProduct!(psi, H_start::MPO, mpo::iMPO; nsweeps=10)
    #H_start is the initial finite size MPO
    #used to initialize the enviroment to minic iDMGR growth steps
    # #left enviroment
    psi_left = psi
    site_psi = isiteinds(psi_left)
    site_mpo = isiteinds(H_start)
    Nsite = length(mpo)
    N_start_size = Int(length(H_start) / 2)
    m = Int(N_start_size / Nsite)
    #for simplicity assuming
    #the N_start_size/N should be an integer
    #otherwise therewill be an error
    #matchining the mpo site indices wit mps site indices
    #the psi should be a finite state of one unitcell
    L = ITensor(1.0)
    count = 1
    for i in 1:N_start_size
        if count > Nsite
            count = 1
        end
        psi_sind = site_psi[count]
        mpo_sind = site_mpo[i]
        A = psi_left[count]
        L = ((L * A) * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                        delta(dag(prime(mpo_sind)), prime(psi_sind)))) * dag(prime(A))
        count += 1
    end
    #normalize the L using effective system size to avoid the problem of jordan-block
    # L /= sqrt(2N_start_size)
    #now contract the bulk MPO
    #mpo far left and right link indices
    L_linkind = inds(L)[1]
    site_mpo = isiteinds(mpo.H)
    pmpo_sind = prime.(site_mpo)
    mpo_lind = setdiff(uniqueinds(mpo.H[1], mpo.H[2]), [dag(site_mpo[1]), pmpo_sind[1]])[1]
    mpo_rind = setdiff(uniqueinds(mpo.H[end], mpo.H[end-1]), [dag(site_mpo[end]), pmpo_sind[end]])[1]
    #match mpo link ind
    replaceind!(L, L_linkind, dag(mpo_lind))
    #align the bulk mpo site indes
    for i in 1:Nsite
        replaceind!(mpo.H[i], dag(site_mpo[i]), dag(site_psi[i]))
        replaceind!(mpo.H[i], prime(site_mpo[i]), prime(site_psi[i]))
    end
    current_length = 2N_start_size # 2 for rightenvrioment grows from the right site
    for i in 1:nsweeps
        for j in 1:Nsite
            psi_sind = site_psi[j]
            mpo_sind = site_mpo[j]
            A = psi_left[j]
            L = ((L * A) * mpo.H[j]) * dag(prime(A))
        end
        replaceind!(L, mpo_rind, dag(mpo_lind))
        #renormalize the L 
        current_length += 2Nsite
    end
    #align hbulk inds
    mpo.L0 = L
    return current_length
end
function initializeMPORightProduct!(psi, H_start::MPO, mpo::iMPO; nsweeps=10)
    #H_start is the initial finite size MPO
    #used to initialize the enviroment to minic iDMGR growth steps
    psi_left = psi
    # #left enviroment
    site_psi = isiteinds(psi_left)
    site_mpo = isiteinds(H_start)
    Nsite = length(mpo)
    N_h0 = length(H_start)
    N_start_size = Int(length(H_start) / 2)
    m = Int(N_start_size / Nsite)
    #for simplicity assuming
    #the N_start_size/N should be an integer
    #otherwise therewill be an error
    #matchining the mpo site indices wit mps site indices
    #first for boundary part
    #first prepare the left initial tensor
    L = ITensor(1.0)
    count = Nsite
    for i in N_h0:-1:N_start_size+1
        if count == 0
            count = Nsite
        end
        psi_sind = site_psi[count]
        mpo_sind = site_mpo[i]
        A = psi_left[count]
        L = ((L * A) * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                        delta(dag(prime(mpo_sind)), prime(psi_sind)))) * dag(prime(A))
        count -= 1
    end
    #normalize the L using effective system size to avoid the problem of jordan-block
    # L /= sqrt(2N_start_size)
    #now contract the bulk MPO
    #mpo far left and right link indices
    L_linkind = inds(L)[1]
    site_mpo = isiteinds(mpo.H)
    pmpo_sind = prime.(site_mpo)
    mpo_lind = setdiff(uniqueinds(mpo.H[1], mpo.H[2]), [dag(site_mpo[1]), pmpo_sind[1]])[1]
    mpo_rind = setdiff(uniqueinds(mpo.H[end], mpo.H[end-1]), [dag(site_mpo[end]), pmpo_sind[end]])[1]
    #match mpo link ind
    replaceind!(L, L_linkind, dag(mpo_rind))
    #align the bulk mpo site indes
    current_length = 2N_start_size # 2 for rightenvrioment grows from the right site
    for i in 1:nsweeps
        for j in Nsite:-1:1
            psi_sind = site_psi[j]
            mpo_sind = site_mpo[j]
            A = psi_left[j]
            L = ((L * A) * mpo.H[j]) * dag(prime(A))
        end
        replaceind!(L, mpo_lind, dag(mpo_rind))
        #renormalize the L 
        # L *= sqrt(current_length) / sqrt(current_length + 2Nsite)
        current_length += 2Nsite
    end
    mpo.R0 = L
    return current_length
end
