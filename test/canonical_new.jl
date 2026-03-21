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
