
function mpo_env_linkinds(psi::MPS, P::iMPO)
    N = P.nunitcell
    lind = commonind(P.L0, P.H[1])
    rind = commonind(P.R0, P.H[N])
    return lind, rind
end
function central_site_problem(psi::MPS, P::iMPO; lambda=nothing)
    #calculate the Lambda
    #first, construct the effective H
    #psi is used to find the correct indices
    N = P.nunitcell
    # PH = P.L0 * P.R0
    lind = commonind(psi[1], P.L0)
    rind = commonind(psi[N], P.R0)
    if isnothing(lind)
        return 1, ITensor(1.0)
    end
    if isnothing(lambda)
        lambda = random_itensor(lind, rind)
    end
    # @show lind,rind
    function central_product(v)
        Pv = P.L0 * v * P.R0
        return noprime(Pv)
    end

    lind, rind = mpo_env_linkinds(psi, P)
    replaceind!(P.R0, rind, dag(lind)) 
    #make sure P.R0 and P.L0 share the same link index
    vals, vecs = eigsolve(
        x -> central_product(x),
        lambda,
        1,
        :SR;
        ishermitian=true,
        tol=1e-14,
        krylovdim=20,
        maxiter=1,
        verbosity=0
    )
    #undo
    replaceind!(P.R0, dag(lind), rind)

    return vals[1], vecs[1]
end
#using Eq19 to approximate the left/right canoncial form
function left_canonical_svd(psi0::MPS, S0::ITensor)
    #using one step svd and trough s 
    psi = copy(psi0)
    Nsite = length(psi0)
    #far left and far right indices
    site_inds = isiteinds(psi)
    rind = uniqueind(dag(S0), psi[Nsite])
    psi[Nsite] = psi[Nsite]*dag(S0)
    #replace the right indices of psi[end] back to new one)
    rnew = settags(new_ind(rind), tags(rind))
    replaceind!(psi[Nsite], rind, rnew)

    
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    # pl = prime(lind)
    #the initial transformation matrix
    L_ini = ITensor(1.0)
    for j in 1:Nsite
        A = L_ini * psi[j]
        if j != Nsite
            linds = uniqueinds(A, psi[j+1])
            ltags = tags(commonind(A, psi[j+1]))
        else
            linds = setdiff(inds(A), [rind])
            ltags = tags(rind)
        end
        if j != Nsite
            Q, L_ini = factorize(A, linds; tags=ltags, which_decomp="qr")
        else
            Q, S, L_ini = svd(A, linds)
            S = pseudo_id(copy(S))
            #through away the singular matrix
            Q = Q * S * L_ini
        end
        psi[j] = Q
    end
    #match L_ini indices
    return psi
end
function right_canonical_svd(psi0::MPS, S0::ITensor)
    #using one step svd and trough s 
    psi = copy(psi0)
    Nsite = length(psi0)
    #far left and far right indices
    site_inds = isiteinds(psi)
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    
    lind = uniqueind(dag(S0), psi[1])
    psi[1] = psi[1]*dag(S0)
    lnew = settags(new_ind(lind), tags(lind))
    replaceind!(psi[1], lind, lnew)

    
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    #the initial transformation matrix
    L_ini = ITensor(1.0)
    for j in Nsite:-1:1
        A = L_ini * psi[j]
        if j != 1
            linds = uniqueinds(A, psi[j-1])
            ltags = tags(commonind(A, psi[j-1]))
        else
            linds = setdiff(inds(A), [lind])
            ltags = tags(lind)
        end
        if j != 1
            Q, L_ini = factorize(A, linds; tags=ltags, which_decomp="qr")
        else
            Q, S, L_ini = svd(A, linds)
            S = pseudo_id(copy(S))
            #through away the singular matrix
            Q = Q * S * L_ini
        end
        psi[j] = Q
    end
    #replace the left indices of psi[1] back to new one)
    rnew = settags(new_ind(lind), tags(lind))
    replaceind!(psi[1], lind, rnew)
    return psi
end
