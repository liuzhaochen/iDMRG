#a poor man's vumps
include("vumps_ini.jl")
include("vumps_dmrg.jl")
function vumps_replaceinds!(mpo, psi_left, psi_right, psi, vumps, S0, left_to_right)
    Nsite = length(mpo)
    site_inds = isiteinds(psi_left)
    lind_psi_l = setdiff(uniqueinds(psi_left[1], psi_left[2]), site_inds)[1]
    lind_p = commonind(psi_left[Nsite], mpo.L0)


    site_inds = isiteinds(psi_right)
    rind_p = commonind(psi_right[1], mpo.R0)
    rind_psi_l = setdiff(uniqueinds(psi_right[Nsite], psi_right[Nsite-1]), site_inds)[1]
    #modify the indices of central tensor
    site_inds = isiteinds(psi)
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]

    # replaceind!(psi_left[1], lind_psi_l, dag(lind_p))
    # replaceind!(psi_left[Nsite], lind_p, dag(lind))

    # replaceind!(psi_right[1], rind_p, dag(rind))
    # replaceind!(psi_right[Nsite], rind_psi_l, dag(rind_p))


    replaceind!(psi[1], lind, dag(lind_p))
    replaceind!(psi[Nsite], rind, dag(rind_p))

    replaceind!(vumps.C[1], lind, dag(lind_p))
    replaceind!(vumps.C[end], rind, dag(rind_p))

    replaceind!(vumps.psi_l[1], lind, dag(lind_p))
    replaceind!(vumps.psi_l[end], lind_p, dag(lind))
    replaceind!(vumps.C[end], lind_p, dag(lind))

    replaceind!(vumps.psi_r[end], rind, dag(rind_p))
    replaceind!(vumps.psi_r[1], rind_p, dag(rind))
    replaceind!(vumps.C[1], rind_p, dag(rind))
    # replaceind!(vumps.U_L, lind_p, dag(lind_psi_l))
    # replaceind!(vumps.U_R, rind_p, dag(rind_psi_l))
    replaceind!(S0, rind, dag(rind_p))
    replaceind!(S0, lind, dag(lind_p))
end
function vumps_initializeIMPO!(psi::MPS, H_ini::MPO, mpo::iMPO, vumps::vumps_canonical, left_to_right;
    S0=nothing, expansion=true, ini_r=false, ini_l=false, err=0, outputlevel=0,
    kwargs...)
    #no need to find left/right canoncial form if the initial state is produc state
    #return mixed form psi
    #get miaxed psi
    N = length(psi)
    mpo.niter = 1
    mpo.lpos = 0
    mpo.rpos = length(mpo) + 1
    psi_left, psi_right = vumps_canonical_form(vumps)
    mpo_env!(psi_left, psi_right, S0, H_ini, mpo; kwargs..., ini_l, ini_r, outputlevel,
        tol=max(1e-12, err))
    mpo.LR = Vector{ITensor}(undef, length(mpo))
    vumps_replaceinds!(mpo, psi_left, psi_right, psi, vumps, S0, left_to_right)
    # psi_left = nothing
    # psi_right = nothing
    return psi
end
function vumps(ipsi::iMPS, mpo::iMPO; nstep_max, maxdims, cutoff=1, observer=NoObserver(), env_dim=5, global_update=true,
    eigsolve_krylovdim=30, eigsolve_maxiter=200, obs=nothing, write_when_maxdim_exceeds=nothing, algorithm="vumps",
    tol=(x -> max(1e-12, x / 100)), kwargs...)
    #note, this method does not work for pure product state
    #using iDMRG to prepare initial state
    if algorithm == "vumps"
        return vumps_sequential(ipsi::iMPS, mpo::iMPO; nstep_max, maxdims, cutoff, observer, env_dim, global_update,
            eigsolve_krylovdim=30, eigsolve_maxiter=200, obs, write_when_maxdim_exceeds,
            tol, kwargs...)
    else
        return vumps_parallel(ipsi::iMPS, mpo::iMPO; nstep_max, maxdims, cutoff, observer, env_dim, global_update,
            eigsolve_krylovdim=30, eigsolve_maxiter=200, obs, write_when_maxdim_exceeds,
            tol, kwargs...)
    end
end
function vumps_sequential(ipsi::iMPS, mpo::iMPO; nstep_max, maxdims, cutoff, observer=NoObserver(), env_dim=5, global_update=true,
    eigsolve_krylovdim=30, eigsolve_maxiter=200, obs=nothing, write_when_maxdim_exceeds=nothing,
    tol=(x -> max(1e-12, x / 100)), kwargs...)
    Nt = length(mpo)
    nsweeps = 1
    psi = ipsi.psi
    H_ini = mpo.H0

    S0 = ipsi.S0
    # psi, err = initializeIMPO!(psi, H_ini, mpo, S0=S0)
    vumps = vumps_canonical(Nt)
    psi, err = vumps_canonical_form(psi, S0, vumps)
    psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0,
        ini_r=true, ini_l=true, err=1e-12, env_dim)

    eng_density = 0
    if isnothing(obs)
        obs = local_step_checkdone(; eng_tol=1e-12)
    end
    isdone = false
    if mpo.nsite != 1
        error("only support singe site version")
    end
    solver = vumps_dmrg
    allowed_keys = (:expansion, :eigsolve_maxiter, :bond_maxiter)
    kwargs = filter_kwargs(kwargs, allowed_keys)
    eng = 0
    eng_c = 0
    step = 0
    for j in 1:nstep_max
        #nsteps = global step
        #nstep: number of local steps
        maxdim = maxdims[min(j, length(maxdims))]
        left_to_right = isodd(j)
        order = left_to_right ? (1:Nt) : (Nt:-1:1)
        eng_tol = tol(err)
        @printf "Canoncial Error at step %i :%.2E\n" step err
        flush(stdout)
        for b in order
            step += 1
            #substract environment energy
            eng_c, S0 = central_site_problem(psi, mpo, lambda=S0)
            energyMPOSubtraction!(mpo, eng_c / Nt)
            eng, psi, S0, err = solver(mpo, psi; step, vumps, left_to_right, S0,
                poi=b, nsweeps, maxdim, cutoff, eigsolve_krylovdim, eigsolve_maxiter, observer=obs, write_when_maxdim_exceeds, eigsolve_tol=eng_tol,
                kwargs...)
            #undo environment energy subtract in hamiltonian
            energyMPOSubtraction!(mpo, -eng_c / Nt)
            isdone = checkdone!(observer; eng_density=eng / Nt, step=(0, step), psi, S0)
            isdone && break
            #reinitialize the enviroment with updated psi_left and psi_right (global update)
            if ((b == Nt && left_to_right) || (b == 1 && !left_to_right))
                F = vumps_S_matrix_overlap(S0, vumps)
                @printf "Overlap Error at step %i :%.2E\n" step F
                if global_update
                    psi, err = vumps_canonical_form(psi, S0, vumps; poi=b)
                else
                    err = max(err, F)
                end
            end
            ini_l = false
            ini_r = false
            # if haskey(kwargs, :expansion)
            #     if kwargs[:expansion] && ((b == Nt && left_to_right) || (b == 1 && !left_to_right))
            #         ini_l = !left_to_right
            #         ini_r = left_to_right
            #     end
            # end
            psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0, err=tol(err / 100), ini_l, ini_r)
        end
        # GC.gc(true)
        isdone && break
    end
    ipsi.psi = psi
    ipsi.S0 = S0
    return ipsi
end
function vumps_parallel(ipsi::iMPS, mpo::iMPO; nstep_max, maxdims, cutoff, observer=NoObserver(), env_dim=5, global_update=true,
    eigsolve_krylovdim=30, eigsolve_maxiter=200, obs=nothing, write_when_maxdim_exceeds=nothing,
    tol=(x -> max(1e-12, x / 100)), kwargs...)
    Nt = length(mpo)
    nsweeps = 1
    psi = ipsi.psi
    H_ini = mpo.H0

    S0 = ipsi.S0
    # psi, err = initializeIMPO!(psi, H_ini, mpo, S0=S0)
    vumps = vumps_canonical(Nt)
    psi, err = vumps_canonical_form(psi, S0, vumps)
    psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0,
        ini_r=true, ini_l=true, err=1e-12, env_dim)

    eng_density = 0
    if isnothing(obs)
        obs = local_step_checkdone(; eng_tol=1e-12)
    end
    isdone = false
    if mpo.nsite != 1
        error("only support singe site version")
    end
    solver = vumps_dmrg_parallel
    allowed_keys = (:expansion, :eigsolve_maxiter, :bond_maxiter)
    kwargs = filter_kwargs(kwargs, allowed_keys)
    eng = 0
    eng_c = 0
    step = 0
    F = 0
    for j in 1:nstep_max
        #nsteps = global step
        #nstep: number of local steps
        maxdim = maxdims[min(j, length(maxdims))]
        left_to_right = isodd(j)
        order = left_to_right ? (1:Nt) : (Nt:-1:1)
        eng_tol = tol(err)
        @printf "error and 1-F at step %i :%.2E %.2E\n" step err F
        flush(stdout)
        step += 1
        #substract environment energy
        eng_c, S0 = central_site_problem(psi, mpo, lambda=S0)
        #we could use S0 to replace C[1] or C[N+1] and psi[1] or psi[N]
        psi = vumps_update!(left_to_right, S0, psi, vumps)
        energyMPOSubtraction!(mpo, eng_c / Nt)
        eng, psi, S0, err = solver(mpo, psi; step, vumps, left_to_right, S0,
            nsweeps, maxdim, cutoff, 
            eigsolve_krylovdim, 
            eigsolve_maxiter, 
            observer=obs, 
            write_when_maxdim_exceeds, 
            eigsolve_tol=eng_tol,
            kwargs...)
        #undo environment energy subtract in hamiltonian
        energyMPOSubtraction!(mpo, -eng_c / Nt)
        isdone = checkdone!(observer; eng_density=eng / Nt, step=(0, step), psi, S0)
        isdone && break
        #reinitialize the enviroment with updated psi_left and psi_right (global update)
        F = vumps_S_matrix_overlap(S0, vumps)
        # err = sqrt(2F)
        ini_l = false
        ini_r = false
        psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0, err=tol(err / 100), ini_l, ini_r)
        # GC.gc(true)
        isdone && break
    end
    ipsi.psi = psi
    ipsi.S0 = S0
    return ipsi
end
