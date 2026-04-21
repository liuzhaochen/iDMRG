#a poor man's vumps

function vumps_initializeIMPO!(psi::MPS, H_ini::MPO, mpo::iMPO, left_to_right; S0=nothing, expansion=true,
    kwargs...)
    #no need to find left/right canoncial form if the initial state is produc state
    #return mixed form psi
    #get miaxed psi
    N = length(psi)
    mpo.niter = 1
    mpo.lpos = 0
    mpo.rpos = length(mpo) + 1
    psi_left, err_l = left_canonical_svd(psi, S0)
    psi_right, err_r = right_canonical_svd(psi, S0)
    err = max(err_l, err_r)
    @printf "Canoncial Error :%.2E\n" err
    flush(stdout)
    #using previous L0, R0 as initial
    ini_l = false
    ini_r = false
    if expansion
        ini_l = left_to_right ? false : true
        ini_r = left_to_right ? true : false
    end
    mpo_env!(psi_left, psi_right, S0, H_ini, mpo; kwargs..., ini_l, ini_r, outputlevel=0,
        tol=max(1e-12,1e-3*err))
    Nsite = length(mpo)
    lind_p = commonind(psi_left[Nsite], mpo.L0)
    rind_p = commonind(psi_right[1], mpo.R0)
    #modify the indices of central tensor
    site_inds = isiteinds(psi)
    lind = setdiff(uniqueinds(psi[1], psi[2]), site_inds)[1]
    rind = setdiff(uniqueinds(psi[Nsite], psi[Nsite-1]), site_inds)[1]
    replaceind!(psi[1], lind, dag(lind_p))
    replaceind!(psi[Nsite], rind, dag(rind_p))
    replaceind!(S0, lind, dag(lind_p))
    replaceind!(S0, rind, dag(rind_p))
    psi_left = nothing
    psi_right = nothing
    mpo.LR = Vector{ITensor}(undef, length(mpo))
    return psi, err
end
function vumps(ipsi::iMPS, mpo::iMPO; nstep_max, maxdims, cutoff, observer=NoObserver(),
    eigsolve_krylovdim=20, eigsolve_maxiter=100, obs=nothing, write_when_maxdim_exceeds=nothing, kwargs...)
    Nt = length(mpo)
    swap_poi = iseven(Nt) ? Int(Nt / 2) : Int(Nt / 2 + 1 / 2)
    sites = isiteinds(mpo.H)
    #solve central site problem to get S0
    #for product state as initial state
    #the enviroment does not have links connect to mps
    #note, this method does not work for pure product state
    #using iDMRG to prepare initial state
    nsweeps = 1
    psi = ipsi.psi
    H_ini = mpo.H0

    S0 = ipsi.S0
    psi, err = initializeIMPO!(psi, H_ini, mpo, S0=S0)
    eng_tol = err / 100
    eng_density = 0
    if isnothing(obs)
        obs = local_step_checkdone(; eng_tol)
    end
    #using sweeps system for global steps
    isdone = false
    solver = mpo.nsite == 1 ? vumps_dmrg3S : error("only support singe site version")
    allowed_keys = mpo.nsite == 1 ? (:expansion, :eigsolve_maxiter) : (:eigsolve_maxiter,)
    kwargs = filter_kwargs(kwargs, allowed_keys)
    for j in 1:nstep_max
        #nsteps = global step
        #nstep: number of local steps
        maxdim = maxdims[min(j, length(maxdims))]
        #solve the central site problem
        eng_c, S0 = central_site_problem(psi, mpo, lambda=S0)
        #substract environment energy
        energyMPOSubtraction!(mpo, eng_c / Nt)
        left_to_right = isodd(j)
        #return psi_c as part of the canonical form
        eng, psi_c, C = solver(mpo, psi; step=j, left_to_right, nsweeps, maxdim, cutoff, eigsolve_krylovdim, eigsolve_maxiter, observer=obs, write_when_maxdim_exceeds, eigsolve_tol=eng_tol,
            kwargs...)

        #undo environment energy subtract in hamiltonian
        energyMPOSubtraction!(mpo, -eng_c / Nt)
        #now reinitial the MPO env and get maixed form psi from C
        S0 = dag(C) * S0
        psi, err = vumps_initializeIMPO!(psi_c, H_ini, mpo, left_to_right; S0)
        eng_tol = err / 100
        isdone = checkdone!(observer; energy_density=eng / Nt, step=(0, j), psi, S0)
        isdone && break
        GC.gc(true)
    end
    ipsi.psi = psi
    ipsi.S0 = S0
    return ipsi
end
