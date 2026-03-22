# 2509.06241
# this paper proposed an alternative method for iDMRG
# performing test here
# enviroment
using ITensors
using ITensorMPS
using ITensorMPS: AbstractProjMPO
using ITensorMPS: OneITensor
using KrylovKit: eigsolve, schursolve, Arnoldi
mutable struct myMPO <: AbstractProjMPO
    niter::Int
    lpos::Int
    rpos::Int
    nsite::Int
    nunitcell::Int
    H::MPO #this should be the bulk MPO
    L0::ITensor
    R0::ITensor
    LR::Vector{ITensor}
end
function circshift(psi::MPS, shift::Int)
    N = length(psi)
    tensor = [psi[i] for i in 1:N]
    circshift!(tensor, shift)
    return MPS(tensor)
end
function circshift(mpo::MPO, shift::Int)
    N = length(mpo)
    tensor = [mpo[i] for i in 1:N]
    circshift!(tensor, shift)
    return MPO(tensor)
end
function new_ind(id)
    id_new = Index(id.space)
    if dir(id_new) != dir(id)
        id_new = dag(id_new)
    end
    return id_new
end
function ITensorMPS.lproj(P::myMPO)
    (P.lpos <= 0) && return P.L0
    return P.LR[P.lpos]
end
function ITensorMPS.rproj(P::myMPO)
    (P.rpos >= length(P) + 1) && return P.R0
    return P.LR[P.rpos]
end
function update_env!(poi_swap::Int, P::myMPO, H0, psi::MPS)
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
        L = L * psi[ll+1] * H0[ll+1] * dag(prime(psi[ll+1]))
        ll += 1
    end
    P.L0 = L
    #for the right 
    rl = N + 1
    R = P.niter > 0 ? P.R0 : OneITensor()
    while rl > k + 1
        R = R * psi[rl-1] * H0[rl-1] * dag(prime(psi[rl-1]))
        # @show rl-1
        rl -= 1
    end
    P.R0 = R
    return nothing
end
function swap_mps!(Lambda, Lambda_odd, sites_old, sites_new, psi, swap_poi::Int, P::myMPO)
    #using the swap MPS method to grow the system
    N = P.nunitcell
    sites_old = copy(sites_old)
    #for swap_poi = 2
    #swap A1,A2,A3,A4,A5
    #to
    #A3,A4,A5, A1, A2
    psi = circshift(psi, -swap_poi)
    circshift!(sites_old, -swap_poi)
    #replace indices
    #set current singular value to 1
    #instead of inverting the S matrix
    #using SVD to update left haft
    #Lambda*A3*A4*A5 S^-1 A1,A2*Lambda
    psi[1] = psi[1] * Lambda
    psi[N] = psi[N] * Lambda
    for i in 1:N
        replaceind!(psi[i], sites_old[i], sites_new[i])
    end
    #Lambda*A3*A4*A5S^dag = USV
    #then left part is UV
    #here A_new = UV≊ A*S^-1
    #we could define the error as 
    #norm(A_new*S - A)
    # 1-inner(A_new*S, A)=1-inner(A_new*S, A)
    #perform QR
    site_inds = sites_new
    if isproduct(psi)
        return psi
    end
    L = ITensor(1.0)
    Nl = N - swap_poi
    #far left indices
    lind = setdiff(uniqueinds(psi[1], psi[2]), [sites_new[1]])[1]#
    #far right indices
    rind = commonind(psi[Nl], dag(Lambda_odd))
    error = ITensor(1.0)
    for j in 1:Nl
        if j != Nl
            A = psi[j] * L
        else
            A = psi[j] * L * dag(Lambda_odd)
        end
        if j != Nl
            linds = uniqueinds(A, psi[j+1])
            ltags = tags(commonind(A, psi[j+1]))
        else
            rind = commonind(A, psi[Nl+1])
            linds = setdiff(inds(A), [rind])
            ltags = tags(rind)
        end
        if j != Nl
            Q, L = qr(A, linds)
        else
            Q, S, L_ini = svd(A, linds)
            S = pseudo_id(S)
            #through away the singular matrix
            Q = Q * S * L_ini
        end
        #calculate error 
        site_id = site_inds[j]
        if j != Nl
            # error = error *psi[j] *prime(dag(Q), !site_id)
            error = error * Q * dag(psi[j])
        else
            error = error * Q * Lambda_odd * dag(psi[j])
        end
        psi[j] = Q
    end
    error = 1 - error[]
    @show error
    return psi
end
function central_product(v, delta_ten, P::myMPO)
    Pv = P.L0 * v * delta_ten * P.R0
    return noprime(Pv)
end
function mpo_env_linkinds(psi::MPS, P::myMPO)
    N = P.nunitcell
    ind_psi = collect(inds(psi[1]))
    append!(ind_psi, dag.(prime.(ind_psi)))
    lind = setdiff(inds(P.L0), ind_psi)[1]

    ind_psi = collect(inds(psi[N]))
    append!(ind_psi, dag.(prime.(ind_psi)))
    rind = setdiff(inds(P.R0), ind_psi)[1]
    return lind, rind
end
function new_site_inds(sites_odd)
    phy = string(collect(tags(sites_odd[1]))[1])
    N = length(sites_odd)
    withqn = hasqns(sites_odd[1]) ? true : false

    sites_new = siteinds(phy, N, conserve_qns=withqn)
    return sites_new
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
function insert_sites!(sites_odd, sites_new, poi_swap, psi::MPS, P::myMPO)
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
    P.H = circshift(P.H, -poi_swap)
    #now modify the link indices make sure mpo can be connected
    replaceind!(P.H[N-poi_swap], mpo_link_rind, dag(mpo_link_lind))
    sites_odd = circshift!(sites_odd, -poi_swap)
    for i in 1:N
        replaceind!(P.H[i], dag(sites_odd[i]), dag_site[i])
        replaceind!(P.H[i], prime(sites_odd[i]), prim_site[i])
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
function update_psi!(swap_poi::Int, psi0, Nt)
    #construct infinite MPS from finite MPS
    #break the bond and modify the incies
    N = length(psi0)
    Nf = swap_poi
    orthogonalize!(psi0, Nf)
    #get the U,S,V
    A = psi0[Nf]
    B = psi0[Nf+1]
    rinds = uniqueinds(A, B)
    ltags = tags(commonind(A, B))
    Ua, S, V, spec = svd(A, rinds; lefttags=ltags)
    psi0[Nf] = Ua
    lind = commonind(Ua, S)
    lind_new = settags(lind, tags(lind))
    #replace S by 1
    #to exactly match the indices
    psi0[Nf+1] = V * psi0[Nf+1]
    # rind = commonind(psi0[Nf+1], psi0[Nf])
    # new_lind = new_ind(lind)
    # new_lind = settags(new_lind, "link,n=$Nt")
    replaceind!(psi0[Nf], lind, lind_new)
    # replaceind!(psi0[Nf+1], rind, dag(new_lind))
    new_rind = commonind(psi0[Nf+1], S)
    return [lind_new, dag(new_rind)], S
end
function new_psi!(ind, sites, psi0, P::myMPO)
    #construct infinite MPS from finite MPS
    #construct random MPS as initial state
    lind = ind[1]
    new_lind = ind[2]
    psi = MPS(P.nunitcell)
    dim0 = dim(lind)
    for i in 1:P.nunitcell
        new_ind = i == P.nunitcell ? new_lind : combinedind(combiner(lind, sites[i]))
        #modify dim
        if i != P.nunitcell
            target_dim = min(dim0, dim(new_ind))
            ratio = min(dim0, dim(new_ind)) / dim(new_ind)
            # @show ratio, dim0
            if hasqns(new_ind)
                new_idx = Pair{QN,Int64}[]
                for sp in new_ind.space
                    push!(new_idx, sp[1] => max(ceil(Int, sp[2] * ratio), 1))
                end
                new_idx = Index(new_idx)
            else
                new_idx = Index(target_dim)
            end
            if dir(new_idx) != dir(new_ind)
                new_idx = dag(new_idx)
            end
            new_idx = settags(new_idx, "link, n=$i")
        else
            new_idx = new_ind
        end
        psi[i] = random_itensor(dag(lind), new_idx, sites[i])
        lind = new_idx
    end
    return psi
end
function isiteinds(psi)
    #for open boundary psi
    #using a different method of siteinds
    idx = []
    N = length(psi)
    for i in 1:N
        ids = inds(psi[i])
        for id in ids
            # @show tags(id)
            phy = string(collect(tags(id))[2])
            if phy == "Site"
                push!(idx, noprime(id))
            end
        end
    end
    return unique(idx)
end
include("canonical_form.jl")
function iMPO(H_bulk, Nuc::Int)
    mpo = myMPO(0, 0, Nuc + 1, 2, Nuc, H_bulk,
        ITensor(1.0), ITensor(1.0), Vector{ITensor}(undef, Nuc))
    return mpo
end
function initializeIMPO(psi, H_ini, mpo::myMPO; nsweeps=10, S0=nothing)
    #no need to find left/right canoncial form if the initial state is produc state
    #return mixed form psi
    if isproduct(psi)
        initializeMPOLeftProduct(psi, H_ini, mpo; nsweeps)
        initializeMPORightProduct(psi, H_ini, mpo; nsweeps)
    else
        psi_left = left_canonical_svd(psi, S0)
        psi_right = right_canonical_svd(psi, S0)

        Nx = initializeMPOLeft(psi_left, H_ini, mpo; nsweeps)
        #psi_left = Rinv*AAA*R
        initializeMPORight(psi_right, H_ini, mpo; nsweeps)
        Nsite = length(mpo)
        lind_p = commonind(psi_left[Nsite], mpo.L0)
        rind_p = commonind(psi_right[1], mpo.R0)
        #modify the indices of central tensor
        site_inds = isiteinds(psi)
        lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
        rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
        replaceind!(psi[1], lind, dag(lind_p))
        replaceind!(psi[Nsite], rind, dag(rind_p))
        replaceind!(S0, lind, dag(lind_p))
        replaceind!(S0, rind, dag(rind_p))
        #accordingly, we should change S0 inds
        # psi = mixedForm(psi, L, R)
    end
    mpo.niter = 1
    mpo.lpos = 0
    mpo.rpos = length(mpo) + 1
    return psi
end
function iDMRG(psi::MPS, mpo::myMPO; nsteps, nsweeps, maxdim, cutoff, H_ini)
    Nt = length(mpo)
    swap_poi = iseven(Nt) ? Int(Nt / 2) : Int(Nt / 2 + 1 / 2)
    sites = isiteinds(mpo.H)
    #solve central site problem to get S0
    #for product state as initial state
    #the enviroment does not have links connect to mps
    psi = initializeIMPO(psi, H_ini, mpo)
    S0 = ITensor(1.0)
    # vals, S0 = central_site_problem(psi, mpo)
    eng_density = 0
    nstep = 50
    Nx = Nt
    #sometimes we may want to reinitialize the mpo
    len_glob = length(nsteps)
    for s in 1:len_glob
        #nsteps = global step
        nstep = nsteps[s]
        for i in 1:nstep
            eng, psi = dmrg(mpo, psi; nsweeps, maxdim, cutoff, eigsolve_krylovdim=20, eigsolve_maxiter=1)
            eng_density = eng / Nt
            if i > 1
                @show i, Nx
                @show eng_density
            end
            if i == nstep
                break
            end
            begin
                ind, S = update_psi!(swap_poi, psi, Nt)
                energyMPOSubtraction!(mpo, eng_density)
                update_env!(swap_poi, mpo, mpo.H, psi)
                energyMPOSubtraction!(mpo, -eng_density)
                sites_new = new_site_inds(sites)
                psi = swap_mps!(S, S0, sites, sites_new, psi, swap_poi, mpo)
                insert_sites!(sites, sites_new, swap_poi, psi, mpo)
                sites = sites_new
                #update swap poi
                swap_poi = Nt - swap_poi
                if i > 1
                    lambdamodule(S0, S)
                end
                S0 = S
                Nx += Nt
            end
        end
        # psi = imps_periodic_form(psi, S0)
        if s != len_glob
            psi = initializeIMPO(psi, H_ini, mpo; nsweeps=100, S0=S0)
            # vals, S0 = central_site_problem(psi, mpo;lambda=S0)
            # @show vals
        end
    end
    return psi
end
function lambdamodule(l1, l2)
    size1 = size(l1, 1)
    size2 = size(l2, 1)
    largel1 = Float64[]
    largel2 = Float64[]
    for i = 1:size1
        push!(largel1, l1[i, i])
    end
    sort!(largel1)
    for i = 1:size2
        push!(largel2, l2[i, i])
    end
    sort!(largel2)
    ove = 0
    for i in 1:min(size1, size2)
        ove += largel1[i] * largel2[i]
    end
    @show ove
end
