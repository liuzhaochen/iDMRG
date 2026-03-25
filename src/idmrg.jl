
function initializeIMPO!(psi::MPS, H_ini::MPO, mpo::iMPO; nsweeps=10, S0=nothing,
    kwargs...)
    #no need to find left/right canoncial form if the initial state is produc state
    #return mixed form psi
    if isproduct(psi)
        initializeMPOLeftProduct!(psi, H_ini, mpo; nsweeps)
        initializeMPORightProduct!(psi, H_ini, mpo; nsweeps)
    else
        psi_left = left_canonical_svd(psi, S0)
        psi_right = right_canonical_svd(psi, S0)
        mpo_env!(psi_left, psi_right, S0, H_ini, mpo; kwargs...)

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
        #accordingly, we should change S0 inds
        # psi = mixedForm(psi, L, R)
    end
    mpo.niter = 1
    mpo.lpos = 0
    mpo.rpos = length(mpo) + 1
    return psi
end
mutable struct local_step_checkdone
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
        return true
    end
    # Otherwise, update last_energy and keep going
    o.last_energy = energy
    return false
end
function ITensorMPS.measure!(o::local_step_checkdone;kwargs...)
    return nothing
end
function idmrg(ipsi::iMPS, mpo::iMPO; nstep_max, nsteps, nsweeps, maxdim, cutoff,
    eigsolve_krylovdim=10, tol=1e-12, eng_tol=1e-10, obs=nothing)
    Nt = length(mpo)
    swap_poi = iseven(Nt) ? Int(Nt / 2) : Int(Nt / 2 + 1 / 2)
    sites = isiteinds(mpo.H)
    #solve central site problem to get S0
    #for product state as initial state
    #the enviroment does not have links connect to mps
    psi = ipsi.psi
    H_ini = mpo.H0
    psi = initializeIMPO!(psi, H_ini, mpo)
    S0 = ipsi.S0
    S = S0
    eng_density = 0
    Nx = Nt
    len_glob = length(nsteps)
    if isnothing(obs)
        obs = local_step_checkdone(; eng_tol)
    end
    #using sweeps system for global steps
    for s in 1:nstep_max
        #nsteps = global step
        nstep = nsteps[min(s, length(nsteps))]
        for i in 1:nstep
            eng, psi = dmrg(mpo, psi; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer=obs)
            eng_density = eng / Nt
            if i > 1
                overlap = lambdamodule(S0, S)
                @printf "======================================\n"
                @printf "iDMRG global step: %i local step %i\n" s i
                @printf "Energy density: %s\n" eng_density
                @printf "Bond matrix overlap: %s\n" overlap
                @printf "=====================================\n"
            end
            if i == nstep
                break
            end
            begin
                S = update_psi!(swap_poi, psi)
                energyMPOSubtraction!(mpo, eng_density)
                update_env!(swap_poi, mpo, mpo.H, psi)
                energyMPOSubtraction!(mpo, -eng_density)
                sites_new = new_site_inds(sites)
                psi, error = swap_mps!(S, S0, sites, sites_new, psi, swap_poi, mpo)
                swap_MPO!(sites, sites_new, swap_poi, psi, mpo)
                sites = sites_new
                #update swap poi
                swap_poi = Nt - swap_poi
                S0 = S
            end
        end
        if s != len_glob
            psi = initializeIMPO!(psi, H_ini, mpo; S0, tol)
            obs.last_energy = 0
            #solve the central site problem
            # vals, S0 = central_site_problem(ipsi.psi, mpo)
        end
    end
    ipsi.psi = psi
    ipsi.S0 = S0
    return ipsi
end
