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
    lambda = copy(lambda)
    lam_dim = size(lambda, 1)
    for i in 1:lam_dim
        lambda[i, i] = inv(lambda[i, i])
    end
    return lambda
end
function mixedForm(psi0::MPS, lambda0, lambda_l, lambda_r)
    psi = copy(psi0)
    # lambda = copy(lambda0)
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
function normalizeIMPS(psi0::MPS, lambda0::ITensor)

    psi = copy(psi0)
    lambda = copy(lambda0)
    Nsite = length(psi0)
    #far left and far right indices
    site_inds = isiteinds(psi)
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    #inverse matrix
    lam_dim = size(lambda, 1)
    for i in 1:lam_dim
        lambda[i, i] = inv(lambda[i, i])
    end
    #update the far right tensor
    psi[Nsite] = psi[Nsite] * dag(lambda)
    new_lind = settags(new_ind(lind), tags(lind))
    replaceind!(psi[Nsite], dag(lind), new_lind)
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    #change indices back to original
    # replaceind!(psi[Nsite], dag(lind), rind)
    #then solve the norm of transfer matix
    # TM = ITensor(1.0)
    # for i in 1:Nsite
    #     TM *= psi[i] * dag(prime(psi[i], !site_inds[i]))
    # end
    #solve eigenvlaue problem
    #instead of full decomposition
    #use eigensolve instead
    # eig, vec = eigen(TM, [lind, prime(dag(lind))], [rind, prime(dag(rind))])
    # @show L * delta(dag(rind), prime(rind))
    # eta = 0
    # for i in 1:size(eig, 1)
    #     if norm(eig[i, i]) > eta
    #         eta = abs(eig[i, i])
    #     end
    # end
    ini_eig = random_itensor(dag(lind), prime(lind))
    function product(x)
        #sequential apply transfer matrix to vector
        for i in 1:Nsite
            x *= psi[i] * dag(prime(psi[i], !site_inds[i]))
        end
        # TMx = TM * x
        #replace indices
        replaceind!(x, rind, dag(lind))
        replaceind!(x, prime(dag(rind)), prime(lind))
        return x
    end
    eig, vec = eigsolve(x -> product(x), ini_eig, 1, :LM; tol=1e-10, krylovdim=20, maxiter=4, verbosity=0)
    @show eig[1]
    norm_psi_site = (abs(eig[1]))^(1 / (2Nsite))
    for i in 1:Nsite
        psi[i] *= 1 / norm_psi_site
    end

    return psi
end
function left_canonical(psi0::MPS, lambda0::ITensor; nsweeps=80)
    psi = copy(psi0)
    lambda = copy(lambda0)
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
    #first we need to solve the spectrum of psi to renormalize it
    #initial matrix
    pl = prime(lind)
    # L_ini = random_itensor(pl, dag(lind))
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

function right_canonical(psi0::MPS, lambda0::ITensor; nsweeps=80)
    psi = copy(psi0)
    lambda = copy(lambda0)
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
function initializeMPOLeft(psi, lambda0, H_start::MPO, mpo::myMPO; nsweeps=10)
    #H_start is the initial finite size MPO
    #used to initialize the enviroment to minic iDMGR growth steps
    psi_left, lambda = left_canonical(psi, lambda0)
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
        replaceind!(mpo.H[i], dag(site_mpo[i]), site_psi[i])
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
        if i != nsweeps
            replaceind!(L, rind, dag(lind))
            replaceind!(L, dag(prime(rind)), prime(lind))
        end
        replaceind!(L, mpo_rind, dag(mpo_lind))
        #renormalize the L 
        # L *= sqrt(current_length) / sqrt(current_length + 2Nsite)
        current_length += 2Nsite
    end
    #align hbulk inds
    mpo.L0 = L
    return lambda, current_length
end
function initializeMPORight(psi, lambda0, H_start::MPO, mpo::myMPO; nsweeps=10)
    #H_start is the initial finite size MPO
    #used to initialize the enviroment to minic iDMGR growth steps
    psi_left, lambda = right_canonical(psi, lambda0)
    # psi_right, R = right_canonical(psi, lambda0)
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
    for i in 1:Nsite
        replaceind!(mpo.H[i], dag(site_mpo[i]), site_psi[i])
        replaceind!(mpo.H[i], prime(site_mpo[i]), prime(site_psi[i]))
    end
    current_length = 2N_start_size # 2 for rightenvrioment grows from the right site
    for i in 1:nsweeps
        for j in Nsite:-1:1
            psi_sind = site_psi[j]
            mpo_sind = site_mpo[j]
            A = psi_left[j]
            L = L * A * mpo.H[j] * dag(prime(A))
        end
        if i != nsweeps
            replaceind!(L, lind, dag(rind))
            replaceind!(L, dag(prime(lind)), prime(rind))
        end
        replaceind!(L, mpo_lind, dag(mpo_rind))
        #renormalize the L 
        # L *= sqrt(current_length) / sqrt(current_length + 2Nsite)
        current_length += 2Nsite
    end
    mpo.R0 = L
    return lambda, current_length
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
