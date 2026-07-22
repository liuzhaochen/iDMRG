#dmrg 2site

function dmrg2S(
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
    return dmrg2S(H, psi0, sweeps; kwargs...)
end

function dmrg2S(
    PH,
    psi0::MPS,
    sweeps::Sweeps;
    buf=nothing,
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
    expansion_sweeps=2,
)
    psi = copy(psi0)
    N = length(psi)
    if !isortho(psi) || orthocenter(psi) != 1
        psi = orthogonalize!(PH, psi, 1)
    end
    @assert isortho(psi) && orthocenter(psi) == 1

    if !isnothing(write_when_maxdim_exceeds)
        if (maxlinkdim(psi) > write_when_maxdim_exceeds)
            PH = disk(PH; path=write_path)
        end
    end
    PH = position!(PH, psi, 1)
    energy = 0.0
    psi_dim = maxlinkdim(psi)
    spec = nothing
    solver_para = eig_para(eigsolve_tol, eigsolve_krylovdim, eigsolve_maxiter)
    for sw in 1:nsweep(sweeps)
        residual = 0.0
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
            if sw > expansion_sweeps && expansion
                expansion = false
            end
            for ha in 1:2
                left_to_right = isodd(ha)
                order = left_to_right ? (1:N-1) : (N-1:-1:1)
                for b in order #single site 
                    PH = position!(PH, psi, b)
                    phi = psi[b] * psi[b+1]


                    L = lproj(PH)
                    R = rproj(PH)
                    H0 = PH.H[b]
                    H1 = PH.H[b+1]
                    energy, phi, err = two_site_eig(phi, L, R, H0, H1, buf; solver_para)
                    residual = max(residual, err)
                    target_dim = expansion ? maxdim(sweeps, sw) : psi_dim
                    ortho = left_to_right ? "left" : "right"
                    spec = replacebond!(
                        PH,
                        psi,
                        b,
                        phi;
                        maxdim=target_dim,
                        mindim=mindim(sweeps, sw),
                        cutoff=cutoff(sweeps, sw),
                        eigen_perturbation=nothing,
                        ortho,
                        normalize=true,
                        which_decomp,
                        svd_alg
                    )
                    maxtruncerr = max(maxtruncerr, spec.truncerr)

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
        end
        if outputlevel >= 1
            @printf(
                "After sweep %d energy=%s  maxlinkdim=%d residual =%.2E maxerr=%.2E time=%.3f\n",
                sw,
                energy,
                maxlinkdim(psi),
                residual,
                maxtruncerr,
                sw_time
            )
            flush(stdout)
        end
        isdone = ITensorMPS.checkdone!(observer; energy, psi, sweep=sw, outputlevel)
        isdone && break
    end
    disk_cleanup!(PH)
    return (energy, psi)
end
