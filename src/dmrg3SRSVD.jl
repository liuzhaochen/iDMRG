function dmrg3SRSVD(
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
    exp = get(kwargs, :expansion, true)
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim...)
    setmindim!(sweeps, mindim...)
    setcutoff!(sweeps, cutoff...)
    setnoise!(sweeps, noise...)
    return dmrg3SRSVD(H, psi0, sweeps; kwargs...)
end
function dmrg3SRSVD(
    PH,
    psi0::MPS,
    sweeps::Sweeps;
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
    expansion_sweeps = 2,
)
    psi = copy(psi0)
    N = length(psi)
    if !isortho(psi) || orthocenter(psi) != 1
        psi = orthogonalize!(PH, psi, 1)
    end
    @assert isortho(psi) && orthocenter(psi) == 1

    if !isnothing(write_when_maxdim_exceeds)
        if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
           (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
            PH = disk(PH; path=write_path)
        end
    end
    PH = position!(PH, psi, 1)
    energy_0 = 0.0
    energy = 0.0
    energy_i = 0.0
    spec = nothing
    Nexp = 0
    alg = "global_krylov"
    for sw in 1:nsweep(sweeps)
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
            #from left to right
            left_to_right = true
            if sw > expansion_sweeps && expansion
                expansion = false
            end
            for (b, ha) in sweepnext(N, ncenter=1) #single site 
                PH = position!(PH, psi, b)
                phi = psi[b]
                if b == N && left_to_right
                    @goto Boundary
                end
                if b == 1 && !left_to_right
                    @goto Boundary
                end
                # 5-7 with expansion this gives wrong results?
                L = lproj(PH)
                R = rproj(PH)
                ## allocate relevant tensor
                cache = lanczo_cache()
                H0 = PH.H[b]
                ## using in-place contract! in mpo_product
                vals, vecs = eigsolve(
                    x -> mpo_product(L, R, H0, cache, x),
                    # PH,
                    phi,
                    1,
                    eigsolve_which_eigenvalue;
                    ishermitian,
                    tol=eigsolve_tol,
                    krylovdim=eigsolve_krylovdim,
                    maxiter=eigsolve_maxiter,
                    verbosity=eigsolve_verbosity,
                    eager=eigsolve_krylovdim > 5,
                )
                energy = vals[1]
                phi = vecs[1]

                poi = b - 1
                if left_to_right
                    poi = b + 1
                end
                A = phi
                B = psi[poi]
                if !expansion
                    @goto QR_Norm
                end
                # @timeit timer "dmrg: 1-site-SVD" begin
                rinds = uniqueinds(A, B)
                ltags = tags(commonind(A, B))
                Ua, S, V, spec = svd(A, rinds; lefttags=ltags,
                    maxdim=maxdim(sweeps, sw),
                    mindim=mindim(sweeps, sw),
                    cutoff=cutoff(sweeps, sw),
                )
                # @show commoninds(Ua,phi)
                psi[b] = Ua * S
                psi[poi] = V * B
                # end
                maxtruncerr = max(maxtruncerr, spec.truncerr)
                phi = psi[b]
                if maxtruncerr > 1e-10 && adjust_alpha
                    alpha = max(alpha_min, maxtruncerr) #max(1e-4, maxtruncerr)
                end
                # @timeit timer "dmrg: Expansion" begin
                #a different way for expansion
                #using random matrix
                LR = nothing
                W = PH.H[b]
                com_ind = commonind(psi[poi], phi)
                w_ind = commonind(PH.H[poi], W)

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
                LR = nothing
                if left_to_right
                    LR = lproj(PH)
                else
                    LR = rproj(PH)
                end
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
                for it_qr in 1:rsvd_power_iteration
                    Q = (dag(M) * Q) * M
                    Q, _ = qr(Q, cR)
                end
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
                P_phi_com_idx = commonind(phi, psi[poi])
                #then expand the phi and psi[poi] tensor
                exp_indx = uniqueind(P, phi)
                #then expand the phi
                A, sA = directsum(P => exp_indx, phi => P_phi_com_idx; tags=tags(com_ind))

                #now, expand the next tensor with zero tensors
                out_idx = uniqueinds(psi[poi], phi)
                com_idx = commonind(psi[poi], phi)
                zero_ten = ITensor(dag(exp_indx), out_idx)
                B, sB = directsum(zero_ten => dag(exp_indx), psi[poi] => com_idx; tags=tags(com_ind))
                replaceind!(B, sB, sA)
                # end

                @label QR_Norm
                rinds = uniqueinds(A, B)
                ltags = tags(commonind(A, B))
                U, V = factorize(A, rinds; tags=ltags, ortho="left", which_decomp="qr")
                psi[b] = U
                psi[poi] = V * B
                @label next
                if outputlevel >= 2
                    @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
                    @printf(
                        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                        cutoff(sweeps, sw),
                        maxdim(sweeps, sw),
                        mindim(sweeps, sw)
                    )
                    @printf(
                        "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
                    )
                    flush(stdout)
                end
                @label Boundary
                if b == N && left_to_right
                    left_to_right = false
                end
                if b == 1 && !left_to_right
                    left_to_right = true
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
                "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E mixer=%.2E time=%.3f\n",
                sw,
                energy,
                maxlinkdim(psi),
                maxtruncerr,
                alpha,
                sw_time
            )
            flush(stdout)
        end
        isdone = ITensorMPS.checkdone!(observer; energy, psi, sweep=sw, outputlevel)
        isdone && break
    end
    return (energy, psi)
end
