using ITensors
using ITensorMPS
using iDMRG
using iDMRG: iMPS, iMPO, idmrg, vumps, initializeHam
let
    Nuc = 2
    N = 6 * Nuc #initial system size #as a cheating way to get iMPO
    p = 1 / 2
    # p = 1/2
    Ndn = Int(p * Nuc)
    sites = siteinds("S=1/2", N, conserve_qns=true)
    os = OpSum()
    for j = 1:N-1
        os += 0.5, "S+", j, "S-", j + 1
        os += 0.5, "S-", j, "S+", j + 1
        os += "Sz", j, "Sz", j + 1
    end
    H = MPO(os, sites)
    #initial product state within unit cell
    sites_uc = siteinds("S=1/2", Nuc, conserve_qns=true)
    state = ["Up" for n = 1:Nuc]
    state[1:Ndn] .= "Dn"
    psi0 = MPS(sites_uc, state)

    nstep_max = 3
    nsteps = [3 for i in 1:1]
    nsweeps = 5
    maxdims = [100]
    cutoff = [-1.0]
    #site_start: using W[site_start] as iMPO (cheating way)
    h0 = initializeHam(H, Nuc, sites_uc; site_start=6, ham_uc=1)
    # site = 2 for 2-site DMRG
    infMPO = iMPO(H, h0, Nuc; site=2)
    ipsi = iMPS(psi0)
    @time ipsi = idmrg(ipsi, infMPO;
        nstep_max, nsteps, nsweeps, maxdims, cutoff, eigsolve_krylovdim=30, eigsolve_maxiter=100, alpha=1e-4)
    # vumps only accept 1-site 
    infMPO = iMPO(H, h0, Nuc; site=1)
    nstep_max = 50
    maxdims = [100]
    @time ipsi = vumps(ipsi, infMPO; nstep_max, nsteps, cutoff, maxdims, tol=(x -> min(1e-6, max(1e-12, 1e-2 * x))),
        algorithm="vumps2")
    return nothing
end
