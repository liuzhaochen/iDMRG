#using the psi_L to iterative update the right environment
function grad_local_energy(L, ten_zero, deten)
    Ld = L * ten_zero  #* deteni
    Ld = Ld * deten
    return Ld[]
end
function grad_mpo_env!(psi_left::MPS, S0::ITensor, C0, mpo::iMPO, buf; tol=1e-12,
    ini=true, outputlevel=0, replace=false, env_dim=5, gc_dim=10000)
    Nuc = length(mpo)
    #initialize the left and right MPO env
    max_dim = maxlinkdim(psi_left)
    gc = max_dim >= gc_dim

    site_mpo = isiteinds(mpo.H)
    pmpo_sind = prime.(site_mpo)
    mpo_lind = setdiff(uniqueinds(mpo.H[1], mpo.H[2]), [dag(site_mpo[1]), pmpo_sind[1]])[1]
    mpo_rind = setdiff(uniqueinds(mpo.H[end], mpo.H[end-1]), [dag(site_mpo[end]), pmpo_sind[end]])[1]
    for i in 1:2
        allocate_buffer!(buf[i])
    end

    if ini
        mpo.R0 = mpo.R0 * C0 * dag(prime(C0))
    else
        #################################################
        #
        #             Update the Left Enviroment
        #
        ################################################

        site_psi = isiteinds(psi_left)
        lind = setdiff(uniqueinds(psi_left[1], psi_left[2]), site_psi)[1]
        rind = setdiff(uniqueinds(psi_left[end], psi_left[end-1]), site_psi)[1]

        b = uniqueind(S0, psi_left[1]) #right indices
        deten = delta(dag(b), prime(b)) #delta tensor to link the far right
        link_id = commonind(mpo.L0, mpo.H[1])
        ten_zero = ITensor(dag(link_id)) #on-site tensor to extract energy density
        ten_zero[link_id=>1] = 1
        function lproduct(L; rep=true)
            en_density = local_energy(L, S0, ten_zero, deten) / Nuc
            #substract energy from current hamiltonian
            energyMPOSubtraction!(mpo, en_density)
            L = left_TMv(L, buf, lind, rind, mpo_lind, mpo_rind, psi_left, mpo, replace_inds=rep)
            energyMPOSubtraction!(mpo, -en_density)
            return L, en_density
        end
        #perform few power iteration to improve initial guess

        L = mpo.L0
        L, _ = anderson_accelerate(L, lproduct; tol, outputlevel, m=env_dim, gc)
        # mpo.L0 = lproduct(L, rep=replace)[1]
        # replaceind!(L, dag(lind), rind)
        # replaceind!(L, prime(lind), dag(prime(rind)))
        mpo.L0 = L
    end


    #################################################
    #
    #             Right Enviroment
    #
    ################################################
    site_psi = isiteinds(psi_left)
    lind = setdiff(uniqueinds(psi_left[1], psi_left[2]), site_psi)[1]
    rind = setdiff(uniqueinds(psi_left[end], psi_left[end-1]), site_psi)[1]
    deten = delta(rind, dag(prime(rind)))

    link_id = commonind(mpo.R0, mpo.H[end])
    ten_zero = ITensor(dag(link_id))
    ten_zero[link_id=>end] = 1
    function rproduct(L; rep=true)
        en_density = grad_local_energy(L, ten_zero, deten) / Nuc
        #substract energy from current hamiltonian
        energyMPOSubtraction!(mpo, en_density)
        L = right_TMv(L, buf, lind, rind, mpo_lind, mpo_rind, psi_left, mpo, replace_inds=rep)
        energyMPOSubtraction!(mpo, -en_density)
        return L, en_density
    end
    L = mpo.R0
    L, _ = anderson_accelerate(L, rproduct; tol, outputlevel, m=env_dim, gc)
    # mpo.R0 = rproduct(L, rep=replace)[1]
    # replaceind!(L, dag(rind), lind)
    # replaceind!(L, prime(rind), dag(prime(lind)))
    mpo.R0 = L
    for i in 1:2
        free_buffer!(buf[i])
    end
    return nothing
end


