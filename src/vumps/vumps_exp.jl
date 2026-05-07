using Base: info_color
#solve and expand the bond_dim
function vumps_dmrg(
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
    return vumps_dmrg(H, psi0, sweeps; kwargs...)
end
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
    R = (((R * phi) * H) * prime(dag(phi)))
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
    L = lproj(PH)
    R = rproj(PH)
    #allocate relevant tensor
    Lv = L * phi
    LHv = Lv * PH.H[PH.lpos+1]
    H0 = PH.H[PH.lpos+1]
    #using in-place contract! in mpo_product
    vals, vecs, info = eigsolve(
        x -> mpo_product(L, R, Lv, LHv, H0, x),
        # PH,
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
function vumps_dmrg(
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
    is_boundary = false

    if (left_to_right && poi == N) || (!left_to_right && poi == 1)
        is_boundary = true
    end
    sw_time = @elapsed begin
        # if !isnothing(write_when_maxdim_exceeds)
        #     if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
        #        (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
        #         PH = disk(PH; path=write_path)
        #     end
        # end
        # for b in order
        b = poi
        #as every time, the env is recalculated, no need to store every LR here
        PH = vumps_position!(PH, psi, b)
        energy_A, phi, residual = vumps_site_solve(PH, psi[b]; residual, eigsolve_tol, eigsolve_krylovdim, eigsolve_maxiter)
        poi_l = b + dx

        energy0 = energy_A
        if is_boundary && bond_maxiter != 1
            # @goto Next
        end
        for i in 1:bond_maxiter
            #two steps for bond matrix
            #such that L*C_2 = A = C_1 * R
            energy, C_1 = vumps_bond_right_solve(b, PH, vumps.psi_r, vumps.C[b]; eigsolve_tol)
            vumps.C[b] = C_1
            eng_2, C_2 = vumps_bond_left_solve(b, PH, vumps.psi_l, vumps.C[b+1]; eigsolve_tol)
            vumps.C[b+1] = C_2

            # A = phi * dag(C_2)
            contract!(vumps.psi_l[b], phi, dag(C_2))
            linds = uniqueinds(phi, dag(C_2))
            U, S, V = svd(vumps.psi_l[b], linds)
            S = pseudo_id(S)
            # A = U * S * V
            # vumps.psi_l[b] = A
            contract!(vumps.psi_l[b], U, S * V)

            # A = phi * dag(C_1)
            contract!(vumps.psi_r[b], phi, dag(C_1))
            linds = uniqueinds(phi, dag(C_1))
            U, S, V = svd(vumps.psi_r[b], linds)
            S = pseudo_id(S)
            # A = U * S * V
            # vumps.psi_r[b] = A
            contract!(vumps.psi_r[b], U, S * V)
            err_bond = abs((energy - energy0) / max(0.1, abs(energy0)))
            energy0 = energy
            if err_bond < eigsolve_tol / 10
                break
            end
        end
        if bond_maxiter == 1
            err_l = 1 - (dag(phi)*vumps.psi_l[b]*vumps.C[b+1])[]
            err_r = 1 - (dag(phi)*vumps.psi_r[b]*vumps.C[b])[]
            err = sqrt(2max(abs(err_l), abs(err_r)))
        end
        # @show err
        @label Next
        if left_to_right && b != N
            psi[b] = copy(vumps.psi_l[b])
            psi[poi_l] = vumps.C[b+1] * psi[poi_l]
        elseif !left_to_right && b != 1
            psi[b] = copy(vumps.psi_r[b])
            psi[poi_l] = vumps.C[b] * psi[poi_l]
        else
            psi[b] = phi
            # if left_to_right
            #     psi[b] = vumps.psi_l[b] * vumps.C[b+1]
            # else
            #     psi[b] = vumps.psi_r[b] * vumps.C[b]
            # end
        end
        if b == N && left_to_right
            # S0, err = vumps_S0_problem_left(psi, vumps, S0, PH; eigsolve_tol)
        end
        if b == 1 && !left_to_right
            # S0, err = vumps_S0_problem_right(psi, vumps, S0, PH; eigsolve_tol)
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
    return (energy_A, psi, S0, err)
end
# function vumps_S0_problem_left(psi, vumps, S0, PH; eigsolve_tol, uv_r=nothing)
#     # for b in order
#     N = length(psi)
#     b = N
#     L = lproj(PH)
#     phi = vumps.psi_l[N]
#     L = ((L * phi) * PH.H[b]) * prime(dag(phi))
#     R = rproj(PH)

#     uv_r = uniqueind(vumps.U_L, vumps.psi_l[end])
#     rind = dag(uniqueind(S0, vumps.C[end]))
#     t_ten = delta(dag(uv_r), rind)
#     err = 0
#     eng = 0
#     for i in 1:10
#         err = vumps_gauge_matrix_left!(psi, vumps, S0; uv_r)
#         uvt = vumps.U_L * t_ten
#         L0 = ((L * prime(dag(uvt))) * uvt)
#         vals, vecs, info = eigsolve(
#             x -> bond_product(L0, R, x),
#             S0,
#             1,
#             :SR;
#             ishermitian=true,
#             tol=eigsolve_tol,
#             krylovdim=30,
#             maxiter=100,
#             eager=true,
#         )
#         S0 = vecs[1]
#         eng_er = abs((eng - vals[1]) / max(0.1, abs(eng)))
#         eng = vals[1]
#         if eng_er < eigsolve_tol
#             break
#         end
#     end
#     return S0, err
# end
# function vumps_S0_problem_right(psi, vumps, S0, PH; eigsolve_tol, uv_l=nothing)
#     # for b in order
#     N = length(psi)
#     b = 1
#     L = lproj(PH)
#     R = rproj(PH)
#     phi = vumps.psi_r[1]
#     R = ((R * phi) * PH.H[b]) * prime(dag(phi))

#     uv_r = uniqueind(vumps.U_R, vumps.psi_r[1])
#     rind = dag(uniqueind(S0, vumps.C[1]))
#     t_ten = delta(dag(uv_r), rind)
#     err = 0
#     eng = 0
#     for i in 1:10
#         err = vumps_gauge_matrix_right!(psi, vumps, S0; uv_l)
#         uvt = vumps.U_R * t_ten
#         R0 = ((R * prime(dag(uvt))) * uvt)
#         vals, vecs, info = eigsolve(
#             x -> bond_product(L, R0, x),
#             S0,
#             1,
#             :SR;
#             ishermitian=true,
#             tol=eigsolve_tol,
#             krylovdim=30,
#             maxiter=100,
#             eager=true,
#         )
#         S0 = vecs[1]
#         # eng_er = abs((eng - vals[1]) / max(0.1, eng))
#         eng_er = abs((eng - vals[1]) / max(0.1, abs(eng)))
#         eng = vals[1]
#         if eng_er < eigsolve_tol
#             break
#         end
#     end
#     return S0, err
# end
