
mutable struct vumps_canonical
    psi_l::MPS
    psi_r::MPS
    # U_L::ITensor
    # U_R::ITensor
    C::Vector{ITensor}
    function vumps_canonical(N)
        return new(MPS(N), MPS(N), Vector{ITensor}(undef, N + 1))
    end
end
function vumps_canonical_form(psi::MPS, S0::ITensor, vumps::vumps_canonical; poi=1)
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
    #update far right bond 
    vumps.psi_l[N] = vumps.psi_l[N] * UV
    vumps.C[N+1] = copy(S0)
    replaceind!(vumps.psi_l[N], rind, rnew)
    replaceind!(vumps.C[N+1], dag(rind), dag(rnew))
    replaceind!(UV, rind, rnew)
    # vumps.U_L = UV
    S = copy(S0)
    replaceind!(S, dag(rind), dag(rnew))
    error_l = 1 - (error_l*UV*S)[]

    #periodic right canoncial form
    A = R_ten_r[1] * dag(S0)
    U, S, V = svd(A, commonind(A, vumps.psi_r[1]))
    S = pseudo_id(S)
    UV = U * S * V
    #update fart right bond 
    vumps.psi_r[1] = vumps.psi_r[1] * UV
    vumps.C[1] = copy(S0)
    replaceind!(vumps.psi_r[1], lind, lnew)
    replaceind!(vumps.C[1], dag(lind), dag(lnew))
    replaceind!(UV, lind, lnew)
    # vumps.U_R = UV
    S = copy(S0)
    replaceind!(S, dag(lind), dag(lnew))
    error_r = 1 - (error_r*UV*S)[]

    #get new mixed canonical tensor
    #fix shallow copy
    # psi = copy(vumps.psi_r)
    for i in 1:poi-1
        psi[i] = copy(vumps.psi_l[i])
    end
    for i in poi:N
        psi[i] = copy(vumps.psi_r[i])
    end
    psi[poi] = psi[poi] * vumps.C[poi]

    R_ten_l = nothing
    R_ten_r = nothing
    return psi, sqrt(2 * max(abs(error_l), abs(error_r)))
end
function vumps_gauge_matrix_left!(psi::MPS, vumps::vumps_canonical, S0::ITensor;
    uv_r=nothing)
    #using psi to update boundary bond tensor
    if isnothing(uv_r)
        # uv_r = uniqueind(vumps.U_L, vumps.psi_l[end])
    end
    rind = dag(uniqueind(S0, vumps.C[end]))
    # N = length(psi)
    # lind = (commonind(psi[end], psi[end-1]), isiteinds(psi)[N])
    # Q, R = factorize(psi[end], lind; tags="cLink,$N", which_decomp="qr")

    # vumps.psi_l[N] = Q
    # vumps.C[N+1] = R
    #we might want to get a new rotation matrix U_L and U_R based on
    #current S0
    A = vumps.C[end] * dag(S0)
    U, S, V = svd(A, rind)
    S = pseudo_id(S)
    UV = U * S * V
    replaceind!(UV, rind, uv_r)
    # @show inds(UV)
    # vumps.U_L = UV
    #calculate canonical error here
    err_r = dag(vumps.C[end]) * UV * delta(dag(uv_r), rind) * S0
    err_r = sqrt(2 * abs(1 - err_r[]))
    return err_r
end
function vumps_gauge_matrix_right!(psi::MPS, vumps::vumps_canonical, S0::ITensor;
    uv_l=nothing)
    #we might want to get a new rotation matrix U_L and U_R based on
    #current S0
    if isnothing(uv_l)
        # uv_l = uniqueind(vumps.U_R, vumps.psi_r[1])
    end
    lind = dag(uniqueind(S0, vumps.C[1]))

    # rind = (commonind(psi[1], psi[2]), isiteinds(psi)[1])
    # Q, R = factorize(psi[1], rind; tags="cLink,1", which_decomp="qr")
    # vumps.psi_r[1] = Q
    # vumps.C[1] = R
    A = vumps.C[1] * dag(S0)
    U, S, V = svd(A, lind)
    S = pseudo_id(S)
    UV = U * S * V
    replaceind!(UV, lind, uv_l)
    # vumps.U_R = UV
    err_r = dag(vumps.C[1]) * UV * delta(dag(uv_l), lind) * S0
    err_r = sqrt(2 * abs(1 - err_r[]))
    return err_r
end
function vumps_canonical_form(vumps::vumps_canonical)
    # psi_left = copy(vumps.psi_l)
    # psi_right = copy(vumps.psi_r)
    psi_left = vumps.psi_l
    psi_right = vumps.psi_r
    # psi_left[end] = psi_left[end] * vumps.U_L
    # psi_right[1] = psi_right[1] * vumps.U_R

    return psi_left, psi_right
end
function vumps_S_matrix_overlap(S0::ITensor, vumps::vumps_canonical)
    lind = uniqueind(vumps.C[1], S0)
    rind = uniqueind(S0, vumps.C[1])
    overlap_1 = (dag(S0)*delta(rind, dag(lind))*vumps.C[1])[]

    lind = uniqueind(vumps.C[end], S0)
    rind = uniqueind(S0, vumps.C[end])
    overlap_2 = (dag(S0)*delta(rind, dag(lind))*vumps.C[end])[]
    return 1 - min(abs(overlap_1), abs(overlap_2))
end
function vumps_update!(l_to_r, S0::ITensor, psi::MPS, vumps::vumps_canonical)
    N = length(psi)
    lind = uniqueind(vumps.C[1], S0)
    rind = uniqueind(S0, vumps.C[1])
    vumps.C[1] = copy(S0)
    replaceind!(vumps.C[1], rind, lind)

    lind = uniqueind(vumps.C[N+1], S0)
    rind = uniqueind(S0, vumps.C[N+1])
    vumps.C[N+1] = copy(S0)
    replaceind!(vumps.C[N+1], rind, lind)

    if l_to_r
        psi[1] = vumps.C[1] * vumps.psi_r[1]
    else
        psi[N] = vumps.C[N+1] * vumps.psi_l[N]
    end
    return psi
end
