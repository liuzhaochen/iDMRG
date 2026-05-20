
function initializeIMPO!(psi::MPS, H_ini::MPO, mpo::iMPO; nsweeps=10, S0=nothing, err_0=nothing,
    kwargs...)
    #no need to find left/right canoncial form if the initial state is produc state
    #return mixed form psi
    mpo.niter = 1
    mpo.lpos = 0
    mpo.rpos = length(mpo) + 1
    if isproduct(psi)
        initializeMPOLeftProduct!(psi, H_ini, mpo; nsweeps)
        initializeMPORightProduct!(psi, H_ini, mpo; nsweeps)
        return psi, 0
    else
        psi_left, err_l = left_canonical_svd(psi, S0)
        psi_right, err_r = right_canonical_svd(psi, S0)
        err = max(err_l, err_r)
        @printf "Canoncial Error :%s\n" err
        flush(stdout)
        err_0 = isnothing(err_0) ? 1e-4 * err : err_0
        mpo_env!(psi_left, psi_right, S0, H_ini, mpo; kwargs..., tol=max(1e-12, err_0))
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
end
mutable struct local_step_checkdone <: ITensorMPS.AbstractObserver
    #check if the local step converged
    #using energy as reference
    energy_tol::Float64
    last_energy::Float64
    function local_step_checkdone(; eng_tol=1e-10)
        return new(eng_tol, 0)
    end
end
function ITensorMPS.checkdone!(o::local_step_checkdone; kwargs...)
    sw = kwargs[:sweep]
    energy = kwargs[:energy]
    if abs(energy - o.last_energy) / abs(energy) < o.energy_tol
        println("Stopping Local DMRG step after sweep $sw")
        flush(stdout)
        o.last_energy = 0
        return true
    end
    # Otherwise, update last_energy and keep going
    o.last_energy = energy
    return false
end
function idmrg(ipsi::iMPS, mpo::iMPO; nstep_max, nsteps, nsweeps, maxdims, cutoff, observer=NoObserver(),
    eigsolve_krylovdim=3, tol=(x->min(1e-5, max(1e-12, x/100))), eng_tol=1e-10, obs=nothing, write_when_maxdim_exceeds=nothing, kwargs...)
    Nt = length(mpo)
    swap_poi = iseven(Nt) ? Int(Nt / 2) : Int(Nt / 2 + 1 / 2)
    sites = isiteinds(mpo.H)
    #solve central site problem to get S0
    #for product state as initial state
    #the enviroment does not have links connect to mps
    psi = ipsi.psi
    H_ini = mpo.H0

    S0 = ipsi.S0
    psi, err = initializeIMPO!(psi, H_ini, mpo, S0=S0, err_0=1e-12)
    S = S0
    eng_density = 0
    if isnothing(obs)
        obs = local_step_checkdone(; eng_tol)
    end
    #using sweeps system for global steps
    isdone = false
    solver = mpo.nsite == 1 ? dmrg3SRSVD : dmrg
    allowed_keys = mpo.nsite == 1 ? (:expansion, :eigsolve_maxiter, :alpha, :expansion_sweeps) : (:eigsolve_maxiter,)
    kwargs = filter_kwargs(kwargs, allowed_keys)
    for s in 1:nstep_max
        #nsteps = global step
        nstep = nsteps[min(s, length(nsteps))]
        if iseven(nstep)
            @printf "Should perform odd number of swaps\n"
            nstep += 1
        end
        #nstep: number of local steps
        maxdim_global = maxdims[min(s, length(maxdims))]

        @printf "======================================\n"
        @printf "iDMRG global step: %i\n" s
        for i in 1:nstep
            #solve the central site problem and update bond operator
            eng_c, S0 = central_site_problem(psi, mpo, lambda=S0)
            # eng_c = 0
            #substract environment energy
            maxdim = maxdim_global[min(i, length(maxdim_global))]
            energyMPOSubtraction!(mpo, eng_c / Nt)
            eng, psi = solver(mpo, psi; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer=obs, write_when_maxdim_exceeds, eigsolve_tol=tol(err),
                kwargs...)
            # if i == 1
            #undo environment energy subtract in hamiltonian
            energyMPOSubtraction!(mpo, -eng_c / Nt)
            # end
            eng_density = (eng + eng_c) / Nt
            eng_c = 0
            @printf "Energy density at step (%i,%i): %s\n" s i eng / Nt
            flush(stdout)
            if isodd(i)
                isdone = checkdone!(observer; eng_density=eng / Nt, step=(s, i), psi, S0)
                isdone && break
            end
            i == nstep && break #without swap operation for the last step
            begin
                S = update_psi!(swap_poi, psi)
                energyMPOSubtraction!(mpo, eng_density)
                update_env!(swap_poi, mpo, mpo.H, psi)
                energyMPOSubtraction!(mpo, -eng_density)
                sites_new = new_site_inds(swap_poi, sites)
                psi, error = swap_mps!(S, S0, sites, sites_new, psi, swap_poi, mpo)
                swap_MPO!(sites, sites_new, swap_poi, psi, mpo)
                sites = sites_new
                #update swap poi
                swap_poi = Nt - swap_poi
                # overlap = lambdamodule(S0, S)
                S0 = S
                #for the case of product state, after swap, we apply one step of random gate to connect to sites
                if s == 1 && i == 1
                    psi = random_gate!(psi)
                end
            end
        end
        isdone && break
        #reinitialize environment
        if s != nstep_max
            psi, err = initializeIMPO!(psi, H_ini, mpo; S0, tol, err_0=1e-12, ini_l=false, ini_r=false)
        end
        # GC.gc(true)
    end
    ipsi.psi = psi
    ipsi.S0 = S0
    return ipsi
end
