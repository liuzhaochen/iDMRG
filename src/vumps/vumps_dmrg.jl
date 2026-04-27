#poor man's vumps
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
    sites = isiteinds(psi)
    maxtruncerr = 0.0
    err_bond = 0.0

    uv_l = uniqueind(vumps.U_R, vumps.psi_r[1])
    uv_r = uniqueind(vumps.U_L, vumps.psi_l[end])

    B = ITensor(1.0)
    sw_time = @elapsed begin
        if !isnothing(write_when_maxdim_exceeds)
            if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
               (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
                PH = disk(PH; path=write_path)
            end
        end
        # for b in order
        b = poi
        poi_l = b + dx
        PH = position!(PH, psi, b)
        energy, phi, residual = vumps_site_solve(PH, psi[b]; residual, eigsolve_tol, eigsolve_krylovdim, eigsolve_maxiter)
        #truncate and expand
        if (left_to_right && b != N) || (!left_to_right && b != 1)
            B = psi[poi_l]
        end
        phi, V, maxtruncerr, alpha = subspace_exp(phi, B, PH, b, poi_l, N, left_to_right;
            sweeps, sw, maxtruncerr, adjust_alpha, alpha, alpha_min, rsvd_qn_min_dim)
        #V contain expaned indices
        if (left_to_right && b != N) || (!left_to_right && b != 1)
            psi[poi_l] = V * psi[poi_l]
        else
            S0 = S0 * dag(V)
        end

        #no need to solver central bond problem; only need two QR and one svd to update psi_l and psi_r
        #for QR to get -L-C-
        #only update psi_l for a left-to-right sweep
        if left_to_right
            linds = commoninds(phi, vumps.psi_l[b])
            L, C1 = qr(phi, linds)
            vumps.psi_l[b] = L
            vumps.C[b+1] = C1
        else
            rinds = commoninds(phi, vumps.psi_r[b])
            R, C2 = qr(phi, rinds)
            vumps.psi_r[b] = R
            vumps.C[b] = C2
        end
        #now, move the central to next position and update the corresponding psi_l or psi_r variationally
        if b != N && left_to_right
            psi[b] = copy(L)
            psi[poi_l] = C1 * psi[poi_l]
            #update psi_l[poi_l]
            C = vumps.C[poi_l+1]
            linds = uniqueinds(psi[poi_l], dag(C))
            A = psi[poi_l] * dag(C)
            U, S, V = svd(A, linds)
            S = pseudo_id(S)
            A = U * S * V
            vumps.psi_l[poi_l] = A
        elseif b != 1 && !left_to_right
            psi[b] = copy(R)
            psi[poi_l] = C2 * psi[poi_l]

            A = psi[poi_l] * dag(vumps.C[poi_l])
            linds = uniqueinds(psi[poi_l], dag(vumps.C[poi_l]))
            U, S, V = svd(A, linds)
            S = pseudo_id(S)
            A = U * S * V
            vumps.psi_r[poi_l] = A
        else
            psi[b] = phi
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
        @printf(
            "Sweep: %i Energy=%s  maxlinkdim=%d residual=%.2E maxerr=%.2E time=%.3f\n",
            step,
            energy / length(psi),
            maxlinkdim(psi),
            residual,
            maxtruncerr,
            sw_time
        )
        flush(stdout)
    end
    isdone = ITensorMPS.checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    return (energy, psi, S0, err)
end
