function vumps_makeL!(P, psi, k)
    ll = P.lpos
    if ll ≥ k
        # Special case when nothing has to be done.
        # Still need to change the position if lproj is
        # being moved backward.
        P.lpos = k
        return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(P)
    while ll < k
        L = L * psi[ll+1] * P.H[ll+1] * dag(prime(psi[ll+1]))
        # P.LR[ll+1] = L
        ll += 1
    end
    P.LR[k] = L
    # Needed when moving lproj backward.
    P.lpos = k
    return P
end
function vumps_makeR!(P, psi, k)
    rl = P.rpos
    if rl ≤ k
        # Special case when nothing has to be done.
        # Still need to change the position if rproj is
        # being moved backward.
        P.rpos = k
        return nothing
    end
    N = length(P.H)
    # Make sure rl is no bigger than `N + 1` for the generic logic below
    rl = min(rl, N + 1)
    R = rproj(P)
    while rl > k
        R = R * psi[rl-1] * P.H[rl-1] * dag(prime(psi[rl-1]))
        # P.LR[rl - 1] = R
        rl -= 1
    end
    P.LR[k] = R
    P.rpos = k
    return R
end
function vumps_position!(P::AbstractProjMPO, psi::MPS, pos::Int)
    #construct left-and right environment
    vumps_makeL!(P, psi, pos - 1)
    vumps_makeR!(P, psi, pos + nsite(P))
    return P
end

function bond_product(L, R, v)
    Pv = L * v * R
    return noprime!(Pv)
end
function vumps_dmrg(
    PH,
    psi0::MPS,
    buf;
    step=1,
    poi=1,
    vumps=nothing,
    bond_maxiter=10,
    left_to_right=true,
    observer=NoObserver(),
    outputlevel=1,
    # eigsolve kwargs
    solver_para,
    kwargs...
)
    psi = copy(psi0)
    N = length(psi)

    # @assert isortho(psi) && orthocenter(psi) == 1
    err = 0.0
    energy = 0.0
    residual = 0.0
    spec = nothing
    sw = 1
    ha = left_to_right ? 1 : 2
    dx = left_to_right ? 1 : -1
    maxtruncerr = 0.0
    err_bond = 0.0
    sw_time = @elapsed begin
        b = poi
        #as every time, the env is recalculated, no need to store every LR here
        PH = vumps_position!(PH, psi, b)
        # energy_A, phi, err = vumps_site_solve(PH, psi[b]; solver_para)

        L = lproj(PH)
        R = rproj(PH)
        H0 = PH.H[PH.lpos+1]
        energy_A, phi, err = single_site_eig(psi[b], L, R, H0, buf; solver_para)
        residual = max(residual, err)
        poi_l = b + dx

        energy0 = energy_A
        for i in 1:bond_maxiter
            #two steps for bond matrix
            #such that L*C_2 = A = C_1 * R
            energy, C_1 = vumps_bond_right_solve(b, PH, vumps.psi_r, vumps.C[b], buf; solver_para)
            vumps.C[b] = C_1
            eng_2, C_2 = vumps_bond_left_solve(b, PH, vumps.psi_l, vumps.C[b+1], buf; solver_para)
            vumps.C[b+1] = C_2

            A = phi * dag(C_2)
            # vumps.psi_l[b] = contract!(vumps.psi_l[b], phi, dag(C_2))
            linds = uniqueinds(phi, dag(C_2))
            U, S, V = svd(A, linds)
            S = pseudo_id(S)
            A = U * S * V
            vumps.psi_l[b] = A
            # vumps.psi_l[b] = contract!(vumps.psi_l[b], U, S * V)

            A = phi * dag(C_1)
            # vumps.psi_r[b] = contract!(vumps.psi_r[b], phi, dag(C_1))
            linds = uniqueinds(phi, dag(C_1))
            U, S, V = svd(A, linds)
            S = pseudo_id(S)
            A = U * S * V
            vumps.psi_r[b] = A
            # vumps.psi_r[b] = contract!(vumps.psi_r[b], U, S * V)
            err_bond = abs((energy - energy0) / max(0.1, abs(energy0)))
            energy0 = energy
            if err_bond < solver_para.tol / 10
                break
            end
        end
        if bond_maxiter == 1
            err_l = 1 - (dag(phi)*vumps.psi_l[b]*vumps.C[b+1])[]
            err_r = 1 - (dag(phi)*vumps.psi_r[b]*vumps.C[b])[]
            err = sqrt(2max(abs(err_l), abs(err_r)))
        end
        if left_to_right && b != N
            psi[b] = copy(vumps.psi_l[b])
            psi[poi_l] = vumps.C[b+1] * psi[poi_l]
        elseif !left_to_right && b != 1
            psi[b] = copy(vumps.psi_r[b])
            psi[poi_l] = vumps.C[b] * psi[poi_l]
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
            "Sweep: %i Energy_bond=%s  maxlinkdim=%d residual=%.2E bond_err=%.2E time=%.3f\n",
            step,
            energy_A / length(psi),
            maxlinkdim(psi),
            residual,
            err_bond,
            sw_time
        )
        flush(stdout)
    end
    isdone = ITensorMPS.checkdone!(observer; energy=energy_A, psi, sweep=sw, outputlevel)
    return (energy_A, psi, err)
end

function vumps_dmrg_parallel(
    PH,
    psi0::MPS,
    buf::Vector{<:AbstractBuffer};
    step=1,
    vumps=nothing,
    bond_maxiter=1,
    observer=NoObserver(),
    outputlevel=1,
    write_when_maxdim_exceeds=nothing,
    write_path=tempdir(),
    # eigsolve kwargs
    solver_para,
    sweeps=1,
    kwargs...
)
    psi = copy(psi0)
    N = length(psi)
    sw = 1
    # @assert isortho(psi) && orthocenter(psi) == 1
    if !isnothing(write_when_maxdim_exceeds)
        if (maxlinkdim(psi) > write_when_maxdim_exceeds)
            PH = disk(PH; path=write_path)
        end
    end
    err = 0.0
    energy_A = 0.0
    for sw in 1:sweeps
        spec = nothing
        maxtruncerr = 0.0
        err_bond = 0.0
        energy = 0.0
        residual = 0.0
        left_to_right = isodd(sw)
        order = left_to_right ? (1:N-1) : (N:-1:2)
        dx = left_to_right ? 1 : -1
        ha = left_to_right ? 1 : 2
        sw_time = @elapsed begin
            err_0 = 0
            for b in order
                PH = position!(PH, psi, b)
                phi = psi[b]

                L = lproj(PH)
                R = rproj(PH)
                H0 = PH.H[PH.lpos+1]
                energy_A, phi, err = single_site_eig(phi, L, R, H0, buf; solver_para)
                residual = max(err, residual)
                poi_l = b + dx
                energy0 = energy_A
                for i in 1:bond_maxiter
                    #two steps for bond matrix
                    #such that L*C_2 = A = C_1 * R
                    energy, C_1 = vumps_bond_right_solve(b, PH, vumps.psi_r, vumps.C[b], buf; solver_para)
                    vumps.C[b] = C_1
                    eng_2, C_2 = vumps_bond_left_solve(b, PH, vumps.psi_l, vumps.C[b+1], buf; solver_para)
                    vumps.C[b+1] = C_2

                    A = phi * dag(C_2)
                    # vumps.psi_l[b] = contract!(vumps.psi_l[b], phi, dag(C_2))
                    linds = uniqueinds(phi, dag(C_2))
                    U, S, V = svd(A, linds)
                    S = pseudo_id(S)
                    A = U * S * V
                    vumps.psi_l[b] = A
                    # vumps.psi_l[b] = contract!(vumps.psi_l[b], U, S * V)

                    A = phi * dag(C_1)
                    # contract!(vumps.psi_r[b], phi, dag(C_1))
                    # vumps.psi_r[b] = contract!(vumps.psi_r[b], phi, dag(C_1))
                    linds = uniqueinds(phi, dag(C_1))
                    U, S, V = svd(A, linds)
                    S = pseudo_id(S)
                    A = U * S * V
                    vumps.psi_r[b] = A
                    err_bond0 = abs((energy - energy0) / max(0.1, abs(energy0)))
                    energy0 = energy
                    if err_bond0 < solver_para.tol
                        err_bond = max(err_bond, err_bond0)
                        break
                    end
                end
                if bond_maxiter == 1
                    err_l = 1 - (dag(phi)*vumps.psi_l[b]*vumps.C[b+1])[]
                    err_r = 1 - (dag(phi)*vumps.psi_r[b]*vumps.C[b])[]
                    err_0 = max(err_0, sqrt(2max(abs(err_l), abs(err_r))))
                end
                if left_to_right && b != N
                    psi[b] = copy(vumps.psi_l[b])
                    psi[poi_l] = vumps.C[b+1] * psi[poi_l]
                elseif !left_to_right && b != 1
                    psi[b] = copy(vumps.psi_r[b])
                    psi[poi_l] = vumps.C[b] * psi[poi_l]
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
            err = err_0
        end
        if outputlevel >= 1
            @printf(
                "Step: %i Sweep: %i Energy_bond=%s  maxlinkdim=%d residual=%.2E err=%.2E time=%.3f\n",
                step,
                sw,
                energy_A / length(psi),
                maxlinkdim(psi),
                residual,
                err,
                sw_time
            )
            flush(stdout)
        end
        isdone = ITensorMPS.checkdone!(observer; energy=energy_A, psi, sweep=sw, outputlevel)
        isdone && iseven(sw) && break
    end
    return (energy_A, psi, err)
end
