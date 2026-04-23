using Base: info_color
#solve and expand the bond_dim
function vumps_dmrg3S(
    H,
    psi0::MPS;
    nsweeps,
    maxdim=ITensorMPS.default_maxdim(),
    mindim=ITensorMPS.default_mindim(),
    cutoff=ITensorMPS.default_cutoff(Float64),
    noise=ITensorMPS.default_noise(),
    kwargs...,
)
    # H = permute(H, (linkind, siteinds, linkind))
    # PH = ProjMPO(0, length(H) + 1, 1, H, Vector{ITensor}(undef, length(H)))
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim...)
    setmindim!(sweeps, mindim...)
    setcutoff!(sweeps, cutoff...)
    setnoise!(sweeps, noise...)
    return vumps_dmrg3S(H, psi0, sweeps; kwargs...)
end
function subspace_exp(A, B, PH, b::Int, poi::Int, N::Int, left_to_right::Bool;
    sweeps, sw, maxtruncerr, adjust_alpha, alpha, alpha_min, rsvd_qn_min_dim)

    LR = nothing
    if left_to_right
        LR = lproj(PH)
        if b == N
            rinds = uniqueinds(A, PH.R0)
            ltags = tags(commonind(A, PH.R0))
        else
            rinds = uniqueinds(A, B)
            ltags = tags(commonind(A, B))
        end
    else
        LR = rproj(PH)
        if b == 1
            rinds = uniqueinds(A, PH.L0)
            ltags = tags(commonind(A, PH.L0))
        else
            rinds = uniqueinds(A, B)
            ltags = tags(commonind(A, B))

        end
    end

    Ua, S, V, spec = svd(A, rinds; lefttags=ltags,
        maxdim=maxdim(sweeps, sw),
        mindim=mindim(sweeps, sw),
        cutoff=cutoff(sweeps, sw),
    )
    phi = Ua * S
    #we can expand to V instead
    # psi[poi] = V * B
    #generating random matrix
    maxtruncerr = max(maxtruncerr, spec.truncerr)
    if maxtruncerr > 1e-10 && adjust_alpha
        alpha = max(alpha_min, maxtruncerr) #max(1e-4, maxtruncerr)
    end
    W = PH.H[b]
    com_ind = commonind(V, phi)
    if left_to_right && b == N
        w_ind = commonind(PH.R0, W)
    elseif !left_to_right && b == 1
        w_ind = commonind(PH.L0, W)
    else
        w_ind = commonind(PH.H[poi], W)
    end

    ind_c = combinedind(combiner(w_ind, com_ind))
    dim_all = dim(ind_c)
    dim_phi = dim(com_ind)
    target_dim = ceil(Int, min(dim_phi, 0.1maxdim(sweeps, sw)))
    ratio = target_dim / dim_all
    new_idx = Pair{QN,Int64}[]
    for sp in ind_c.space
        push!(new_idx, sp[1] => max(ceil(Int, sp[2] * ratio), rsvd_qn_min_dim))
    end
    ind_cnew = Index(new_idx)
    if dir(ind_cnew) == dir(ind_c)
        ind_cnew = dag(ind_cnew)
    end
    rand_ten = random_itensor(ind_cnew, w_ind, com_ind)


    M = (LR * phi) * W
    noprime!(M)
    M = M - (M * dag(Ua)) * Ua
    P = M * rand_ten
    if b == 1 || b == N
        cR = commonind(P, W)
        clk = commonind(P, LR)
        cR = !isnothing(clk) ? [cR, clk] : cR
    else
        cR = [commonind(P, LR), commonind(P, W)]
    end
    Q, _ = qr(P, cR)
    qr_ind = uniqueind(Q, P)
    MQ = M * dag(Q)
    bond_dim = dim(ind_cnew)
    # 10: oversampling parameter 
    expand_dim = bond_dim > 10 ? bond_dim - 10 : bond_dim
    U, _ = factorize(MQ, dag(qr_ind), maxdim=
        min(expand_dim, target_dim),
        ortho="right", which_decomp="svd")
    P = Q * U * alpha
    noprime!(P)

    P_phi_com_idx = commonind(phi, V)
    #then expand the phi and psi[poi] tensor
    exp_indx = uniqueind(P, phi)
    #then expand the phi
    A, sA = directsum(P => exp_indx, phi => P_phi_com_idx; tags=tags(com_ind))
    #instead of expanding the next tensor
    #we expand V matrix 
    #now, expand the next tensor with zero tensors
    out_idx = uniqueinds(V, phi)
    com_idx = commonind(V, phi)
    zero_ten = ITensor(dag(exp_indx), out_idx)
    B, sB = directsum(zero_ten => dag(exp_indx), V => com_idx; tags=tags(com_ind))
    replaceind!(B, sB, sA)
    #construct one identity matrix 
    v_id = delta(com_idx, out_idx)
    return A, B, maxtruncerr, alpha
end
function bond_product(L, R, v)
    Pv = L * v * R
    return noprime!(Pv)
end
function vumps_bond_left_solve(i::Int, PH, psi_l::MPS, C
    ;
    eigsolve_tol=1.0e-10,
    eigsolve_krylovdim=30,
    eigsolve_maxiter=100,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,
    ishermitian=true)
    L = lproj(PH)
    R = rproj(PH)
    H = PH.H[i]
    phi = psi_l[i]
    #update L 
    L = (((L * phi) * H) * prime(dag(phi)))
    vals, vecs, info = eigsolve(
        x -> bond_product(L, R, x),
        C,
        1,
        eigsolve_which_eigenvalue;
        ishermitian,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
        eager=true,
    )
    return vals[1], vecs[1]
end
function vumps_bond_right_solve(i::Int, PH, psi_r::MPS, C;
    eigsolve_tol=1.0e-10,
    eigsolve_krylovdim=30,
    eigsolve_maxiter=100,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,
    ishermitian=true)
    L = lproj(PH)
    R = rproj(PH)
    H = PH.H[i]
    phi = psi_r[i]
    #update R 
    R = (((R * phi) * H )* prime(dag(phi)))
    vals, vecs, info = eigsolve(
        x -> bond_product(L, R, x),
        C,
        1,
        eigsolve_which_eigenvalue;
        ishermitian,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
        eager=true,
    )
    return vals[1], vecs[1]
end
function vumps_site_solve(PH, phi;
    residual,
    eigsolve_tol=1.0e-10,
    eigsolve_krylovdim=30,
    eigsolve_maxiter=100,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,
    ishermitian=true)
    vals, vecs, info = eigsolve(
        PH,
        phi,
        1,
        eigsolve_which_eigenvalue;
        ishermitian,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
        eager=true,
    )
    residual = max(residual, info.normres[1])

    energy = vals[1]
    phi = vecs[1]
    return energy, phi, residual
end
function vumps_dmrg3S(
    PH,
    psi0::MPS,
    sweeps::Sweeps;
    step=1,
    poi=1,
    vumps=nothing,
    bond_maxiter=10,
    S0=nothing,
    left_to_right=true,
    which_decomp=nothing,
    svd_alg=nothing,
    observer=NoObserver(),
    outputlevel=1,
    write_when_maxdim_exceeds=nothing,
    write_path=tempdir(),
    # eigsolve kwargs
    eigsolve_tol=1.0e-10,
    eigsolve_krylovdim=3,
    eigsolve_maxiter=1,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,
    ishermitian=true,
    # rsvd
    rsvd_qn_min_dim=2,
    rsvd_power_iteration=0,
    expansion=true,
    alpha=2e-2,
    alpha_min=1e-8,
    adjust_alpha=true,
)
    psi = copy(psi0)
    N = length(psi)

    # @assert isortho(psi) && orthocenter(psi) == 1
    err = 0.0
    energy = 0.0
    residual = 0.0
    spec = nothing
    sw = 1
    # central_bond_tensor = nothing
    ha = left_to_right ? 1 : 2
    dx = left_to_right ? 1 : -1
    # sites = isiteinds(psi)
    maxtruncerr = 0.0
    err_bond = 0.0
    sw_time = @elapsed begin
        if !isnothing(write_when_maxdim_exceeds)
            if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
               (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
                PH = disk(PH; path=write_path)
            end
        end
        # for b in order
        b = poi
        PH = position!(PH, psi, b)
        energy_A, phi, residual = vumps_site_solve(PH, psi[b]; residual, eigsolve_tol, eigsolve_krylovdim, eigsolve_maxiter)
        poi_l = b + dx

        energy0 = energy
        for i in 1:bond_maxiter
            #two steps for bond matrix
            #such that L*C_2 = A = C_1 * R
            energy, C_1 = vumps_bond_right_solve(b, PH, vumps.psi_r, vumps.C[b]; eigsolve_tol)
            vumps.C[b] = C_1
            eng_2, C_2 = vumps_bond_left_solve(b, PH, vumps.psi_l, vumps.C[b+1]; eigsolve_tol)
            vumps.C[b+1] = C_2

            A = phi * dag(C_2)
            linds = uniqueinds(phi, dag(C_2))
            U, S, V = svd(A, linds)
            S = pseudo_id(S)
            A = U * S * V
            vumps.psi_l[b] = A

            A = phi * dag(C_1)
            linds = uniqueinds(phi, dag(C_1))
            U, S, V = svd(A, linds)
            S = pseudo_id(S)
            A = U * S * V
            vumps.psi_r[b] = A
            err_bond = abs((energy - energy0) / max(0.1, abs(energy0)))
            energy0 = energy
            if err_bond < eigsolve_tol / 10 || i == bond_maxiter
                break
            end
        end
        err_l = 1 - (dag(phi)*vumps.psi_l[b]*vumps.C[b+1])[]
        err_r = 1 - (dag(phi)*vumps.psi_r[b]*vumps.C[b])[]
        err = sqrt(2max(abs(err_l), abs(err_r)))
        # @show err
        if left_to_right && b != N
            psi[b] = copy(vumps.psi_l[b])
            psi[poi_l] = vumps.C[b+1] * psi[poi_l]
        elseif !left_to_right && b != 1
            psi[b] = copy(vumps.psi_r[b])
            psi[poi_l] = vumps.C[b] * psi[poi_l]
        else
            if left_to_right
                # psi[b] = phi
                psi[b] = vumps.psi_l[b] * vumps.C[b+1]
            else
                psi[b] = vumps.psi_r[b] * vumps.C[b]
            end
        end
        if b == N
            S0, err = vumps_S0_problem_left(psi, vumps, S0, PH; eigsolve_tol)
        end
        if b == 1
            S0, err = vumps_S0_problem_right(psi, vumps, S0, PH; eigsolve_tol)
        end
        sweep_is_done = (b == 1 && ha == 2)
        ITensorMPS.measure!(
            observer;
            energy,
            psi,
            projected_operator=PH,
            bond=b,
            sweep=sw,
            half_sweep=ha,
            spec,
            outputlevel,
            sweep_is_done,
        )
    end
    if outputlevel >= 1
        # @printf(
        #     "Sweep: %i Energy=%s  maxlinkdim=%d maxerr=%.2E mixer=%.2E residual=%.2E time=%.3f\n",
        #     step,
        #     energy / length(psi),
        #     maxlinkdim(psi),
        #     maxtruncerr,
        #     alpha,
        #     residual,
        #     sw_time
        # )
        @printf(
            "Sweep: %i Energy_bond=%s  maxlinkdim=%d residual=%.2E bond_err=%.2E time=%.3f\n",
            step,
            energy / length(psi),
            # energy_A / length(psi),
            maxlinkdim(psi),
            residual,
            err_bond,
            sw_time
        )
        flush(stdout)
    end
    isdone = ITensorMPS.checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    return (energy, psi, S0, err)
end
function vumps_central_bonds(left_to_right::Bool, psi::MPS, C::ITensor,
    PH::iMPO, write_when_maxdim_exceeds=4000)
    #solving the corresponding central site problem
    #construct temperal env
    psi = copy(psi)
    N = length(psi)
    order = left_to_right ? (N:-1:1) : (1:N)
    dx = left_to_right ? -1 : 1
    C0 = C
    central_bonds = Vector{ITensor}(undef, N)
    for b in order
        # PH = position!(PH, psi, b)
        # L = lproj(PH)
        # R = rproj(PH)
        #update L
        # L = L * psi[b] * PH.H[b] * prime(dag(psi[b]))
        central_bonds[b] = C0
        A = C0 * psi[b]
        if left_to_right && b == 1
            psi[b] = A
            break
        end
        if !left_to_right && b == N
            psi[b] = A
            break
        end
        #shift bond to next position
        B = psi[b+dx]
        rinds = uniqueinds(A, B)
        ltags = tags(commonind(A, B))
        U, V = factorize(A, rinds; tags=ltags, ortho="left", which_decomp="qr")
        psi[b] = U
        C0 = V
    end
    return psi, central_bonds
end
function vumps_S0_problem_left(psi, vumps, S0, PH; eigsolve_tol)
    # for b in order
    N = length(psi)
    b = N
    L = lproj(PH)
    phi = vumps.psi_l[N]
    L = ((L * phi) * PH.H[b]) * prime(dag(phi))
    R = rproj(PH)

    uv_r = uniqueind(vumps.U_L, vumps.psi_l[end])
    rind = dag(uniqueind(S0, vumps.C[end]))
    t_ten = delta(dag(uv_r), rind)
    err = 0
    eng = 0
    for i in 1:10
        err = vumps_gauge_matrix_left!(psi, vumps, S0)
        uvt = vumps.U_L * t_ten
        L0 = ((L * prime(dag(uvt))) * uvt)
        vals, vecs, info = eigsolve(
            x -> bond_product(L0, R, x),
            S0,
            1,
            :SR;
            ishermitian=true,
            tol=eigsolve_tol,
            krylovdim=30,
            maxiter=100,
            eager=true,
        )
        S0 = vecs[1]
        eng_er = abs((eng - vals[1]) / max(0.1, abs(eng)))
        eng = vals[1]
        if eng_er < eigsolve_tol
            break
        end
    end
    return S0, err
end
function vumps_S0_problem_right(psi, vumps, S0, PH; eigsolve_tol)
    # for b in order
    N = length(psi)
    b = 1
    L = lproj(PH)
    R = rproj(PH)
    phi = vumps.psi_r[1]
    L = ((L * phi) * PH.H[b]) * prime(dag(phi))

    uv_r = uniqueind(vumps.U_R, vumps.psi_r[1])
    rind = dag(uniqueind(S0, vumps.C[1]))
    t_ten = delta(dag(uv_r), rind)
    err = 0
    eng = 0
    for i in 1:10
        err = vumps_gauge_matrix_right!(psi, vumps, S0)
        uvt = vumps.U_R * t_ten
        R0 = ((R * prime(dag(uvt))) * uvt)
        vals, vecs, info = eigsolve(
            x -> bond_product(L, R0, x),
            S0,
            1,
            :SR;
            ishermitian=true,
            tol=eigsolve_tol,
            krylovdim=30,
            maxiter=100,
            eager=true,
        )
        S0 = vecs[1]
        # eng_er = abs((eng - vals[1]) / max(0.1, eng))
        eng_er = abs((eng - vals[1]) / max(0.1, abs(eng)))
        eng = vals[1]
        if eng_er < eigsolve_tol
            break
        end
    end
    return S0, err
end
