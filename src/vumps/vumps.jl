#a poor man's vumps
include("vumps_ini.jl")
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
    replaceind!(vumps.psi_r[end], rind, dag(rind_p))

    replaceind!(vumps.U_L, lind_p, dag(lind_psi_l))
    replaceind!(vumps.U_R, rind_p, dag(rind_psi_l))
    replaceind!(S0, rind, dag(rind_p))
    replaceind!(S0, lind, dag(lind_p))
end
function vumps_initializeIMPO!(psi::MPS, H_ini::MPO, mpo::iMPO, vumps::vumps_canonical, left_to_right;
    S0=nothing, expansion=true, ini_r=false, ini_l=false, err=0,
    kwargs...)
    #no need to find left/right canoncial form if the initial state is produc state
    #return mixed form psi
    #get miaxed psi
    N = length(psi)
    mpo.niter = 1
    mpo.lpos = 0
    mpo.rpos = length(mpo) + 1
    psi_left, psi_right = vumps_canonical_form(vumps)
    flush(stdout)
    mpo_env!(psi_left, psi_right, S0, H_ini, mpo; kwargs..., ini_l, ini_r, outputlevel=0,
        tol=max(1e-12, 1e-2 * err))
    mpo.LR = Vector{ITensor}(undef, length(mpo))
    vumps_replaceinds!(mpo, psi_left, psi_right, psi, vumps, S0, left_to_right)
    psi_left = nothing
    psi_right = nothing
    return psi
end
function vumps(ipsi::iMPS, mpo::iMPO; nstep_max, maxdims, cutoff, observer=NoObserver(),
    eigsolve_krylovdim=30, eigsolve_maxiter=200, obs=nothing, write_when_maxdim_exceeds=nothing,
    tol=(x -> max(1e-14, x / 100)), kwargs...)
    #note, this method does not work for pure product state
    #using iDMRG to prepare initial state
    Nt = length(mpo)
    nsweeps = 1
    psi = ipsi.psi
    H_ini = mpo.H0

    S0 = ipsi.S0
    # psi, err = initializeIMPO!(psi, H_ini, mpo, S0=S0)
    vumps = vumps_canonical(Nt)
    psi, err = vumps_canonical_form(psi, S0, vumps)
    psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0,
        ini_r=true, ini_l=true, expansion=false, err)


    eng_density = 0
    if isnothing(obs)
        obs = local_step_checkdone(; eng_tol=1e-12)
    end
    #using sweeps system for global steps
    isdone = false
    solver = mpo.nsite == 1 ? vumps_dmrg3S : error("only support singe site version")
    allowed_keys = mpo.nsite == 1 ? (:expansion, :eigsolve_maxiter) : (:eigsolve_maxiter,)
    kwargs = filter_kwargs(kwargs, allowed_keys)
    eng = 0
    eng_c = 0
    step = 0
    for j in 1:nstep_max
        #nsteps = global step
        #nstep: number of local steps
        @printf "Canoncial Error :%.2E\n" err
        maxdim = maxdims[min(j, length(maxdims))]

        order = 1:Nt
        eng_tol = tol(err)
        for b in order
            step += 1
            #substract environment energy
            eng_c, S0 = central_site_problem(psi, mpo, lambda=S0)
            energyMPOSubtraction!(mpo, eng_c / Nt)
            eng, psi = solver(mpo, psi; step, vumps,
                poi=b, nsweeps, maxdim, cutoff, eigsolve_krylovdim, eigsolve_maxiter, observer=obs, write_when_maxdim_exceeds, eigsolve_tol=eng_tol,
                kwargs...)
            #undo environment energy subtract in hamiltonian
            energyMPOSubtraction!(mpo, -eng_c / Nt)
            #reinitialize the enviroment with updated psi_left and psi_right
            if b != Nt
                # vumps_gauge_matrix_left!(vumps, S0)
                # vumps_gauge_matrix_right!(vumps, S0)
                psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0, err)
            end
        end

        # for b in order
        #     step += 1
        #     #substract environment energy
        #     eng_c, S0 = central_site_problem(psi, mpo, lambda=S0)
        #     energyMPOSubtraction!(mpo, eng_c / Nt)
        #     eng, psi = solver(mpo, psi; step, vumps,
        #         poi=b, nsweeps, maxdim, cutoff, eigsolve_krylovdim, eigsolve_maxiter, observer=obs, write_when_maxdim_exceeds, eigsolve_tol=eng_tol,
        #         kwargs...)
        #     #undo environment energy subtract in hamiltonian
        #     #for this update, no need to solve central site problem
        #     energyMPOSubtraction!(mpo, -eng_c / Nt)
        #     if b != Nt
        #         psi, err = vumps_canonical_form(psi, S0, vumps; poi=b + 1)
        #         @printf "Canoncial Error :%.2E\n" err
        #         psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0, err)
        #     end
        # end

        #update S0 matrix 
        # eng_c, S0 = central_site_problem(psi, mpo, lambda=S0)
        #update left/right canonical based on current mixed form
        psi, err = vumps_canonical_form(psi, S0, vumps)
        psi = vumps_initializeIMPO!(psi, H_ini, mpo, vumps, true; S0, kwargs..., err)
        isdone = checkdone!(observer; eng_density=eng / Nt, step=(0, j), psi, S0)
        isdone && break
        GC.gc(true)
    end
    ipsi.psi = psi
    ipsi.S0 = S0
    return ipsi
end
