
mutable struct vumps_canonical
    psi_l::MPS
    psi_r::MPS
    U_L::ITensor
    U_R::ITensor
    C::Vector{ITensor}
    function vumps_canonical(N)
        return new(MPS(N), MPS(N), ITensor(1.0), ITensor(1.0), Vector{ITensor}(undef, N + 1))
    end
end
function vumps_canonical_form(psi::MPS, S0::ITensor, vumps::vumps_canonical)
    N = length(psi)
    site_inds = isiteinds(psi)
    #indices of S0
    lind = uniqueind(dag(S0), psi[1])
    lnew = settags(new_ind(lind), tags(lind))
    rind = uniqueind(dag(S0), psi[N])
    rnew = settags(new_ind(rind), tags(rind))


    p_lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    p_rind = setdiff(uniqueinds(psi[N], psi[N-1]), site_inds)[1]
    #the R tensor of R in left canoncial process
    R_ten_l = Vector{ITensor}(undef, N)
    R_ten_r = Vector{ITensor}(undef, N)
    L_ini = ITensor(1.0)
    error_l = ITensor(1.0) #error tensor
    for j in 1:N
        A = L_ini * psi[j]
        if j != N
            linds = uniqueinds(A, psi[j+1])
            ltags = tags(commonind(A, psi[j+1]))
        else
            linds = setdiff(inds(A), [p_rind])
            ltags = tags(p_rind)
        end
        Q, L_ini = factorize(A, linds; tags=ltags, which_decomp="qr")
        vumps.psi_l[j] = Q
        R_ten_l[j] = L_ini
        error_l = (error_l * Q) * dag(psi[j])
    end
    L_ini = ITensor(1.0)
    error_r = ITensor(1.0) #error tensor
    for j in N:-1:1
        A = L_ini * psi[j]
        if j != 1
            linds = uniqueinds(A, psi[j-1])
            ltags = tags(commonind(A, psi[j-1]))
        else
            linds = setdiff(inds(A), [p_lind])
            ltags = tags(p_lind)
        end
        Q, L_ini = factorize(A, linds; tags=ltags, which_decomp="qr")
        vumps.psi_r[j] = Q
        R_ten_r[j] = L_ini
        error_r = (error_r * Q) * dag(psi[j])
    end
    #construct the central bond tensors
    vumps.C[1] = R_ten_r[1]
    vumps.C[N+1] = R_ten_l[N]
    for i in 2:N
        vumps.C[i] = R_ten_l[i-1] * R_ten_r[i]
    end
    #get the boundary transfermation matrix, which change vumps.psi_l into a true
    #periodic left canoncial form
    A = R_ten_l[N] * dag(S0)
    U, S, V = svd(A, commonind(A, vumps.psi_l[N]))
    S = pseudo_id(S)
    UV = U * S * V
    replaceind!(UV, rind, rnew)
    vumps.U_L = UV
    S = copy(S0)
    replaceind!(S, dag(rind), dag(rnew))
    error_l = 1 - (error_l*UV*S)[]

    #periodic right canoncial form
    A = R_ten_r[1] * dag(S0)
    U, S, V = svd(A, commonind(A, vumps.psi_r[1]))
    S = pseudo_id(S)
    UV = U * S * V

    replaceind!(UV, lind, lnew)
    vumps.U_R = UV
    S = copy(S0)
    replaceind!(S, dag(lind), dag(lnew))
    error_r = 1 - (error_r*UV*S)[]

    #get new mixed canonical tensor
    psi = copy(vumps.psi_r)
    psi[1] = psi[1] * R_ten_r[1]

    R_ten_l = nothing
    R_ten_r = nothing
    return psi, max(abs(error_l), abs(error_r))
end
function vumps_canonical_form(vumps::vumps_canonical)
    psi_left = copy(vumps.psi_l)
    psi_right = copy(vumps.psi_r)
    psi_left[end] = psi_left[end] * vumps.U_L
    psi_right[1] = psi_right[1] * vumps.U_R

    return psi_left, psi_right
end
