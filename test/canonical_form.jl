#calculate left and right canonical form
function central_site_problem(psi::MPS, P::myMPO)
    #calculate the Lambda
    #first, construct the effective H
    #psi is used to find the correct indices
    N = P.nunitcell
    Nf = Int(N / 2)
    # PH = P.L0 * P.R0
    lind = commonind(psi[1], P.L0)
    rind = commonind(psi[N], P.R0)
    if isnothing(lind)
        return 1, ITensor(1.0)
    end
    lambda = random_itensor(lind, rind)
    # @show lind,rind

    lind, rind = mpo_env_linkinds(psi, P)
    delta_ten = delta(dag(lind), dag(rind))

    # central_product(lambda, delta_ten, P)
    vals, vecs = eigsolve(
        x -> central_product(x, delta_ten, P),
        lambda,
        1,
        :SR;
        ishermitian=true,
        tol=1e-10,
        krylovdim=20,
        maxiter=1,
        verbosity=0
    )
    return vals[1], vecs[1]
end
function pseudo_inverse(lambda)
    if lambda == ITensor(1.0)
    else
        lambda = copy(lambda)
        lam_dim = size(lambda, 1)
        for i in 1:lam_dim
            val = lambda[i, i]
            lambda[i, i] = abs(val) > 1e-16 ? inv(lambda[i, i]) : 0
        end
    end
    return lambda
end
function pseudo_sqrt_root(lambda)
    if lambda == ITensor(1.0)
    else
        lambda = copy(lambda)
        lam_dim = size(lambda, 1)
        for i in 1:lam_dim
            val = lambda[i, i]
            # @show val
            lambda[i, i] = val > 0 ? sqrt(lambda[i, i]) : 0
        end
    end
    return lambda
end
function mixedForm(psi0::MPS, lambda_l, lambda_r)
    psi = copy(psi0)
    Nsite = length(psi0)
    #far left and far right indices
    site_inds = isiteinds(psi)
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    #inverse matrix
    # lam_dim = size(lambda, 1)
    # for i in 1:lam_dim
    #     lambda[i, i] = inv(lambda[i, i])
    # end
    # psi[Nsite] = psi[Nsite] * dag(lambda)
    #change indices back to original
    replaceind!(psi[Nsite], dag(lind), rind)
    #multiply the left and right L tensor
    psi[1] = psi[1] * lambda_l
    psi[Nsite] *= lambda_r
    return psi
end
function imps_periodic_form(psi0::MPS, lambda0::ITensor)
    #update to periodic form of psi

    psi = copy(psi0)
    lambda = copy(lambda0)
    Nsite = length(psi0)
    #far left and far right indices
    #inverse matrix
    lam_dim = size(lambda, 1)
    for i in 1:lam_dim
        lambda[i, i] = inv(lambda[i, i])
    end
    #update the far right tensor
    psi[Nsite] = psi[Nsite] * dag(lambda)
    return psi
end
function normalizeIMPS(psi0::MPS)
    psi = psi0
    Nsite = length(psi0)
    site_inds = isiteinds(psi)
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    new_lind = settags(new_ind(lind), tags(lind))
    replaceind!(psi[Nsite], dag(lind), new_lind)
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    ini_eig = random_itensor(prime(lind), dag(lind))
    function leftproduct(x)
        #sequential apply transfer matrix to vector
        #find the left fixed point
        for i in 1:Nsite
            x = (x * psi[i]) * dag(prime(psi[i], !site_inds[i]))
        end
        # TMx = TM * x
        #replace indices
        replaceind!(x, rind, dag(lind))
        replaceind!(x, prime(dag(rind)), prime(lind))
        return x
    end
    function rightproduct(x)
        #sequential apply transfer matrix to vector
        #find the right fixed point
        for i in Nsite:-1:1
            x = (x * psi[i]) * dag(prime(psi[i], !site_inds[i]))
        end
        # TMx = TM * x
        #replace indices
        replaceind!(x, lind, dag(rind))
        replaceind!(x, prime(dag(lind)), prime(rind))
        return x
    end
    #left fixed point matrix
    eig, vec = eigsolve(x -> leftproduct(x), ini_eig, 2, :LM; tol=1e-14, krylovdim=20, maxiter=10, verbosity=0)
    norm_psi_site = (abs(eig[1]))^(1 / (2Nsite))
    for i in 1:Nsite
        psi[i] *= 1 / norm_psi_site
    end
    #perform eigen value decouple of vec
    L = vec[1]
    D, U = eigen(L, prime(lind), dag(lind), ishermitian=true)
    #now check square root of D
    D *= sign(D[1, 1])
    D = pseudo_sqrt_root(D)
    dl, dr = uniqueind(D, U), commonind(D, U)
    L = noprime(D * dag(U))
    L_inv = noprime(dag(pseudo_inverse(D)) * U)
    replaceind!(L_inv, lind, dag(rind))
    ind = commonind(L_inv, L)
    new_lind = settags(new_ind(ind), tags(ind))
    replaceind!(L_inv, ind, new_lind)
    #the left canoncial is L*A...A*L_inv
    #and L is used in mixed representation


    #the right fixed point
    ini_eig = random_itensor(dag(rind), prime(rind))
    eig, vec = eigsolve(x -> rightproduct(x), ini_eig, 2, :LM; tol=1e-14, krylovdim=20, maxiter=10, verbosity=0)
    R = vec[1]
    D, U = eigen(R, dag(rind), prime(rind), ishermitian=true)
    D *= sign(D[1, 1])
    D = pseudo_sqrt_root(D)
    dl, dr = uniqueind(D, U), commonind(D, U)
    R = noprime(D * prime(U))
    R_inv = noprime(dag(pseudo_inverse(D)) * dag(prime(U)))
    replaceind!(R_inv, rind, dag(lind))
    ind = commonind(R_inv, R)
    new_lind = settags(new_ind(ind), tags(ind))
    replaceind!(R_inv, ind, new_lind)
    #the right canoncial is R_inv*A...A*R
    return psi, L, L_inv, R, R_inv
end
function left_canonical(L, L_inv, psi0::MPS; nsweeps=20)
    psi = copy(psi0)
    Nsite = length(psi0)
    psi[1] = psi[1] * L
    psi[Nsite] = psi[Nsite] * L_inv
    return psi
end

function right_canonical(R, R_inv, psi0::MPS; nsweeps=20)
    psi = copy(psi0)
    Nsite = length(psi0)
    psi[1] = psi[1] * R_inv
    psi[Nsite] = psi[Nsite] * R
    return psi
end
function left_canonical(psi0::MPS; nsweeps=20)
    psi = copy(psi0)
    Nsite = length(psi0)
    #far left and far right indices
    site_inds = isiteinds(psi)
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    @show lind, rind
    pl = prime(lind)
    L_ini = delta(pl, dag(lind))
    for i in 1:nsweeps
        #perform qr for L*psi
        L0 = L_ini
        for j in 1:Nsite
            A = L_ini * psi[j]
            if j != Nsite
                linds = uniqueinds(A, psi[j+1])
                ltags = tags(commonind(A, psi[j+1]))
            else
                linds = setdiff(inds(A), [rind])
                ltags = tags(rind)
            end
            Q, L_ini = factorize(A, linds; tags=ltags, ortho="left", which_decomp="qr")
            #construct transfer matrix for arnoldi step
            # psi[b] = U
            # psi[poi] = V * B
            if i == nsweeps
                psi[j] = Q
            else
                #normalize L_ini
                L_ini /= norm(L_ini)
            end
        end
        #for next round calculation, we need to shift the indices
        i == 1 && @show inds(L_ini)
        i == 1 && @show array(L_ini)
        i == 1 && @show flux(L_ini)
        replaceind!(L_ini, rind, dag(lind))
        #according to ref, we could do a arnoldi step to improve the L_ini
        # vals, vecs = eigsolve(x->TM*x,)

        #replace indics for future usage
        lind0 = setdiff(inds(L_ini), [dag(lind)])[1]
        # replaceind!(L_ini, lind, pl)
        pl = uniqueind(L0, L_ini)
        da = delta(lind0, dag(pl)) * L0
        inner = (dag(L_ini)*da)[]
        err = abs(inner) - 1.0
        if i > nsweeps - 10
            @show i, err
        end
        # @show commonind(L0, L_ini)
    end
    return psi, L_ini
end

function right_canonical(psi0::MPS; nsweeps=20)
    psi = copy(psi0)
    Nsite = length(psi0)
    #far left and far right indices
    site_inds = isiteinds(psi)
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    #inverse matrix
    # lam_dim = size(lambda, 1)
    # for i in 1:lam_dim
    #     lambda[i, i] = inv(lambda[i, i])
    # end
    # #update the far right tensor
    # psi[Nsite] = psi[Nsite] * dag(lambda)
    # #change indices back to original
    # replaceind!(psi[Nsite], dag(lind), rind)
    #initial matrix
    pr = prime(rind)
    L_ini = delta(dag(rind), pr)
    #sweep from left to right
    for i in 1:nsweeps
        #perform qr for L*psi
        for j in Nsite:-1:1
            A = psi[j] * L_ini
            if j != 1
                linds = uniqueinds(A, psi[j-1])
                ltags = tags(commonind(A, psi[j-1]))
            else
                linds = setdiff(inds(A), [lind])
                ltags = tags(lind)
            end
            Q, L_ini = factorize(A, linds; tags=ltags, ortho="left", which_decomp="qr")
            # psi[b] = U
            # psi[poi] = V * B
            if i == nsweeps
                psi[j] = Q
            else
                #normalize L_ini
                L_ini /= norm(L_ini)
            end
        end
        #for next round calculation, we need to shift the indices
        replaceind!(L_ini, lind, dag(rind))
    end
    return psi, L_ini
end
function initializeMPOLeft(psi_left, H_start::MPO, mpo::myMPO; nsweeps=10)
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
        L = L * A * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                     delta(dag(prime(mpo_sind)), prime(psi_sind))) * dag(prime(A))
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
    current_length = 2N_start_size # 2 for rightenvrioment grows from the right site
    for i in 1:nsweeps
        for j in 1:Nsite
            psi_sind = site_psi[j]
            mpo_sind = site_mpo[j]
            A = psi_left[j]
            L = L * A * mpo.H[j] * dag(prime(A))
        end
        # if i != nsweeps
        replaceind!(L, rind, dag(lind))
        replaceind!(L, dag(prime(rind)), prime(lind))
        # end
        replaceind!(L, mpo_rind, dag(mpo_lind))
        #renormalize the L 
        # L *= sqrt(current_length) / sqrt(current_length + 2Nsite)
        current_length += 2Nsite
    end
    #align hbulk inds
    mpo.L0 = L
    return current_length
end
function initializeMPORight(psi_left, H_start::MPO, mpo::myMPO; nsweeps=10)
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
        L = L * A * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                     delta(dag(prime(mpo_sind)), prime(psi_sind))) * dag(prime(A))
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
    #align the bulk mpo site indes
    # for i in 1:Nsite
    #     replaceind!(mpo.H[i], dag(site_mpo[i]), site_psi[i])
    #     replaceind!(mpo.H[i], prime(site_mpo[i]), prime(site_psi[i]))
    # end
    current_length = 2N_start_size # 2 for rightenvrioment grows from the right site
    for i in 1:nsweeps
        for j in Nsite:-1:1
            psi_sind = site_psi[j]
            mpo_sind = site_mpo[j]
            A = psi_left[j]
            L = L * A * mpo.H[j] * dag(prime(A))
        end
        # if i != nsweeps
        replaceind!(L, lind, dag(rind))
        replaceind!(L, dag(prime(lind)), prime(rind))
        # end
        replaceind!(L, mpo_lind, dag(mpo_rind))
        #renormalize the L 
        # L *= sqrt(current_length) / sqrt(current_length + 2Nsite)
        current_length += 2Nsite
    end
    mpo.R0 = L
    return current_length
end
function energyMPOSubtractionInI(H, en_density)
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

function energyMPOSubtraction!(mpo::myMPO, en_density)
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
function initializeMPOLeftProduct(psi, H_start::MPO, mpo::myMPO; nsweeps=10)
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
        L = L * A * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                     delta(dag(prime(mpo_sind)), prime(psi_sind))) * dag(prime(A))
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
            L = L * A * mpo.H[j] * dag(prime(A))
        end
        replaceind!(L, mpo_rind, dag(mpo_lind))
        #renormalize the L 
        current_length += 2Nsite
    end
    #align hbulk inds
    mpo.L0 = L
    return current_length
end
function initializeMPORightProduct(psi, H_start::MPO, mpo::myMPO; nsweeps=10)
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
        L = L * A * (delta(dag(psi_sind), mpo_sind) * H_start[i] *
                     delta(dag(prime(mpo_sind)), prime(psi_sind))) * dag(prime(A))
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
            L = L * A * mpo.H[j] * dag(prime(A))
        end
        replaceind!(L, mpo_lind, dag(mpo_rind))
        #renormalize the L 
        # L *= sqrt(current_length) / sqrt(current_length + 2Nsite)
        current_length += 2Nsite
    end
    mpo.R0 = L
    return current_length
end
function isproduct(psi::MPS)
    #check whether the psi has not links between unitcell
    ind_1 = inds(psi[1])
    ispro = false
    if length(ind_1) < 3
        ispro = true
    end
    return ispro
end
