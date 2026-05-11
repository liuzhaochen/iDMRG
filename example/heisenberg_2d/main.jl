using MKL
using LinearAlgebra
using Strided
using ITensors
using ITensorMPS
using iDMRG
using HDF5
if Threads.nthreads()>1
    BLAS.set_num_threads(1)
    Strided.set_num_threads(1)
    ITensors.enable_threaded_blocksparse(true)
end
let
    Ny = 6
    Nx = 12

    N = Nx * Ny
    Nuc = 2Ny
    symm = true
    sites = siteinds("S=1/2", N; conserve_qns = symm)
    lattice = square_lattice(Nx, Ny; yperiodic = true)
    os = OpSum()
    for b in lattice
        os += 0.5, "S+", b.s1, "S-", b.s2
        os += 0.5, "S-", b.s1, "S+", b.s2
        os += "Sz", b.s1, "Sz", b.s2
    end
    H = MPO(os, sites)
    #perform DMRG first
    sites_uc = siteinds("S=1/2", Nuc; conserve_qns = symm)
    state = [isodd(n) ? "Up" : "Dn" for n = 1:Ny]
    for i in 1:Ny
        if isodd(i)
            push!(state, "Dn")
        else
            push!(state, "Up")
        end
    end
    psi0 = random_mps(sites_uc, state;linkdims=10)
    # psi0 = MPS(sites_uc, state)
    nsteps = [3 for i in 1:20]
    nstep_max = length(nsteps)
    nsweeps = 4
    maxdims = [512]
    cutoff = [-1.0]

    h0 = iDMRG.initializeHam(H, Nuc, sites_uc; site_start=5*Ny+1, ham_uc=Ny)
    infMPO = iDMRG.iMPO(H, h0, Nuc, site = 2)
    ipsi = iDMRG.iMPS(psi0)
    #E_ref = 0.672788 for W=6
    #−0.672 724 840 92 for vumps paper
    @time ipsi = iDMRG.idmrg(ipsi, infMPO; nstep_max, nsteps, nsweeps, maxdims, cutoff, eigsolve_krylovdim=5)

    
    infMPO = iDMRG.iMPO(H, h0, Nuc, site = 1)
    nsteps = [3]
    nstep_max = 20
    @time ipsi = iDMRG.idmrg(ipsi, infMPO; nstep_max, nsteps, nsweeps, maxdims, cutoff, eigsolve_krylovdim=3, expansion = false)

    nstep_max = 500
    infMPO = iDMRG.iMPO(H, h0, Nuc, site = 1)
    @time ipsi = iDMRG.vumps(ipsi, infMPO; nstep_max, tol = (x->min(1e-4, max(1e-12,x/100))), algorithm = "vumps2")
    # saving mps
    # h5open("./t.hdf5", "w") do f
    #     write(f, "psi", ipsi)
    # end
    # ipsi0 = nothing
    # h5open("./t.hdf5","r") do f
    #     ipsi0 = read(f,"psi", iDMRG.iMPS)
    # end
    return nothing
end

