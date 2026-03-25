
mutable struct iMPS
    psi::MPS
    S0::ITensor
    function iMPS(psi::MPS)
        return new(psi, ITensor(1.0))
    end
end
function swap_mps!(Lambda, Lambda_odd, sites_old, sites_new, psi::MPS, swap_poi::Int, P::iMPO)
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
        return psi, 0
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
    return psi, error
end
function update_psi!(swap_poi::Int, psi0::MPS)
    #construct infinite MPS from finite MPS
    #break the bond and modify the incies
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
    replaceind!(psi0[Nf], lind, lind_new)
    # replaceind!(psi0[Nf+1], rind, dag(new_lind))
    return S
end
