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
function vumps_dmrg3S(
    PH,
    psi0::MPS,
    sweeps::Sweeps;
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
    if left_to_right
        psi = orthogonalize!(PH, psi, 1)
    else
        psi = orthogonalize!(PH, psi, N)
    end
    # @assert isortho(psi) && orthocenter(psi) == 1

    if !isnothing(write_when_maxdim_exceeds)
        if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
           (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
            PH = disk(PH; path=write_path)
        end
    end
    PH = position!(PH, psi, 1)
    energy = 0.0
    residual = 0.0
    spec = nothing
    Nexp = 0
    alg = "global_krylov"
    sw = 1
    order = left_to_right ? (1:N) : (N:-1:1)
    central_bond_tensor = nothing
    ha = left_to_right ? 1 : 2
    sw_time = @elapsed begin
        maxtruncerr = 0.0
        if !isnothing(write_when_maxdim_exceeds) &&
           maxdim(sweeps, sw) > write_when_maxdim_exceeds
            if outputlevel >= 2
                println(
                    "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
                )
            end
            PH = disk(PH; path=write_path)
        end
        for b in order
            PH = position!(PH, psi, b)
            phi = psi[b]
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

            poi = b - 1
            A = phi
            if left_to_right
                poi = b + 1
            end
            B = ITensor(1.0)
            if left_to_right && b == N
            elseif !left_to_right && b == 1
            else
                B = psi[poi]
            end

            if !expansion
                @goto QR_Norm
            end
            A, V, maxtruncerr, alpha = subspace_exp(A, B, PH, b, poi, N, left_to_right; sweeps, sw, maxtruncerr,
                adjust_alpha, alpha, alpha_min, rsvd_qn_min_dim)
            B = V * B
            @label QR_Norm
            if left_to_right && b == N
                psi[b] = A
                central_bond_tensor = B
            elseif !left_to_right && b == 1
                psi[b] = A
                central_bond_tensor = B
            else
                rinds = uniqueinds(A, B)
                ltags = tags(commonind(A, B))
                U, V = factorize(A, rinds; tags=ltags, ortho="left", which_decomp="qr")
                psi[b] = U
                psi[poi] = V * B
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
    end
    if outputlevel >= 1
        @printf(
            "Energy=%s  maxlinkdim=%d maxerr=%.2E mixer=%.2E residual=%.2E time=%.3f\n",
            energy,
            maxlinkdim(psi),
            maxtruncerr,
            alpha,
            residual,
            sw_time
        )
        flush(stdout)
    end
    isdone = ITensorMPS.checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    return (energy, psi, central_bond_tensor)
end
function vumps_central_bond(left_to_right::Bool, psi::MPS, C0::ITensor, PH::iMPO)
    #solving the corresponding central site problem
    #construct temperal env
    N = length(psi)
    b = N
    if left_to_right
        L = lproj(PH)
        R = PH.R0
    else
        L = rproj(PH)
        R = PH.L0
        b = 1
    end
    #update L
    L = L * psi[b] * PH.H[b] * prime(dag(psi[b]))
    #solve the central site 
    function central_product(v)
        Pv = (L * v) * R
        return noprime(Pv)
    end
    vals, vecs = eigsolve(
        x -> central_product(x),
        C0,
        1,
        :SR;
        ishermitian=true,
        tol=1e-14,
        krylovdim=20,
        maxiter=100,
        verbosity=0
    )
    return vals[1], vecs[1]
end
