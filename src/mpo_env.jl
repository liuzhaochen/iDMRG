#using power iteration method to calculate the mpo left/right fixed point
#to accelerate power iteration
#using the anderson method
function anderson_accelerate(L_init, product_func; m=5, tol=1e-12, max_iter=300, outputlevel=1, gc=false)
    Ls = Vector{ITensor}(undef, m)
    Rs = Vector{ITensor}(undef, m)
    G_cache = zeros(m, m)  # inner product of residule <Ri, Rj>

    L = L_init
    # R = copy(L)
    en_prev = 0.0
    #first cache the Ls and Rs vector
    # for i in 1:m
    #     Ls[i] = similar(L)
    #     Rs[i] = similar(L)
    #     Ls[i].=0
    #     Rs[i].=0
    # end
    for j in 1:max_iter
        # --- Power Iteration  ---
        L_next, en = product_func(L)
        # R.=0
        # R.=L_next # 
        # R .-= L
        R = L_next - L
        # err_R = (R*dag(R))[]
        err = abs(en - en_prev) / max(abs(en), 1.0)
        if err < tol
            if outputlevel >= 1
                @printf(
                    "DIIS Env @ %i Energy=%s  err=%.2E\n",
                    j,
                    en,
                    err,
                    # err_R
                )
                # println("Anderson converged at step $j, Energy: $en, err:$err, err_R:$err_R")
                flush(stdout)
            end
            return L_next, en
        end
        if mod(j, 100) == 0
            @printf(
                "DIIS Env @ %i Energy=%s  err=%.2E\n",
                j,
                en,
                err,
            )
            flush(stdout)
        end
        if mod(j, m) == 0 && gc
            GC.gc(true)
        end
        en_prev = en

        # --- update Gram  ---
        curr_idx = (j - 1) % m + 1
        Ls[curr_idx] = L
        Rs[curr_idx] = R

        # update residule matrix
        for i in 1:m
            if isassigned(Rs, i)
                if i == curr_idx
                    # diagonal
                    G_cache[i, i] = real((Rs[i]*dag(Rs[i]))[])
                else
                    # off diagonal
                    val = real((Rs[curr_idx]*dag(Rs[i]))[])
                    G_cache[curr_idx, i] = val
                    G_cache[i, curr_idx] = val
                end
            end
        end

        # --- DIIS Equation ---
        k_active = count(i -> isassigned(Rs, i), 1:m)
        if k_active >= 2
            # subspace
            sub_G = G_cache[1:k_active, 1:k_active]

            # (∑α = 1)
            # [ G  -1 ] [ α ] = [ 0 ]
            # [ -1  0 ] [ λ ] = [ -1 ]
            G_ext = zeros(k_active + 1, k_active + 1)
            G_ext[1:k_active, 1:k_active] = sub_G
            G_ext[1:k_active, k_active+1] .= -1.0
            G_ext[k_active+1, 1:k_active] .= -1.0
            #preventing singular matrix
            # for i in 1:k_active
            #     G_ext[i, i] += 1e-12
            # end

            rhs = zeros(k_active + 1)
            rhs[k_active+1] = -1.0

            try
                # sol = G_ext \ rhs
                invG = pinv(G_ext)
                sol = invG * rhs
                alpha = sol[1:k_active]
                alpha = alpha / sum(alpha)

                # new tensor：L_new = ∑ α_i * (L_i + R_i)
                L_new = alpha[1] * (Ls[1] + Rs[1])
                for i in 2:k_active
                    L_new += alpha[i] * (Ls[i] + Rs[i])
                end
                L = L_new
            catch e
                L = L_next
            end
        else
            L = L_next
        end
    end
    #free the mem
    Ls = nothing
    Rs = nothing
    GC.gc()
    return L, en_prev
end
#the local energy terms in enviroment MPO
function local_energy(L, S0, ten_zero, deten)
    Ld = L * S0 * ten_zero * prime(dag(S0)) * deten
    return Ld[]
end
#one step of TM*v
function left_TMv(L, buf, lind, rind, mpo_lind, mpo_rind, psi_left, mpo; replace_inds=true)
    #apply one TMPO to left environment
    Nsite = length(mpo)
    Lb = to_buffer(L, buf[1])
    #move psi to buffer
    with_alloc_buffer(buf[1]) do
        for j in 1:Nsite
            # psi_sind = site_psi[j]
            # mpo_sind = site_mpo[j]
            A = psi_left[j]
            #cache the Ls
            #intermediate tensor will be on buf2
            Lb = with_alloc_buffer(buf[2]) do
                hb = to_buffer(mpo.H[j], buf[2])
                Ab = to_buffer(A, buf[2])
                L1 = Lb * Ab
                #reset buf1 to release L
                reset!(buf[1])
                L1 * hb * dag(prime(Ab))
            end
            #this move L to buffer of buf[1]
            Lb = move_to_heap(Lb, buf[2])
            #release intermediate alloc on buf2
            reset!(buf[2])
        end
    end
    L = move_to_heap(Lb, buf[1])
    reset!(buf[1])
    if replace_inds
        replaceind!(L, rind, dag(lind))
        replaceind!(L, dag(prime(rind)), prime(lind))
    end
    replaceind!(L, mpo_rind, dag(mpo_lind))
    return L
end
function right_TMv(L, buf, lind, rind, mpo_lind, mpo_rind, psi_left, mpo; replace_inds=true)
    #apply one TMPO to left environment
    Nsite = length(mpo)
    Lb = to_buffer(L, buf[1])
    # site_psi = isiteinds(psi_left)
    with_alloc_buffer(buf[1]) do
        for j in Nsite:-1:1
            # psi_sind = site_psi[j]
            # mpo_sind = site_mpo[j]
            A = psi_left[j]
            Lb = with_alloc_buffer(buf[2]) do
                hb = to_buffer(mpo.H[j], buf[2])
                Ab = to_buffer(A, buf[2])
                L1 = Lb * Ab
                reset!(buf[1])
                L1 * hb * dag(prime(Ab))
            end
            #move ram to buf1
            Lb = move_to_heap(Lb, buf[2])
            reset!(buf[2])
            # L = ((L * A) * mpo.H[j]) * dag(prime(A))
        end
    end
    L = move_to_heap(Lb, buf[1])
    reset!(buf[1])
    # L = copy(Ls[1])
    if replace_inds
        replaceind!(L, lind, dag(rind))
        replaceind!(L, dag(prime(lind)), prime(rind))
    end
    replaceind!(L, mpo_lind, dag(mpo_rind))
    return L
end
function mpo_env!(psi_left::MPS, psi_right::MPS, S0::ITensor, H_ini::MPO, mpo::iMPO, buf; tol=1e-12, ini_l=true,
    ini_r=true, outputlevel=1, replace=false, env_dim=5, gc_dim=10000)
    Nuc = length(mpo)
    #initialize the left and right MPO env
    max_dim = maxlinkdim(psi_left)
    gc = max_dim >= gc_dim
    if ini_l
        initializeMPOLeft!(psi_left, H_ini, mpo)
    end
    if ini_r
        initializeMPORight!(psi_right, H_ini, mpo)
    end
    #bulk MPO for the far left and right link indices
    site_mpo = isiteinds(mpo.H)
    pmpo_sind = prime.(site_mpo)
    mpo_lind = setdiff(uniqueinds(mpo.H[1], mpo.H[2]), [dag(site_mpo[1]), pmpo_sind[1]])[1]
    mpo_rind = setdiff(uniqueinds(mpo.H[end], mpo.H[end-1]), [dag(site_mpo[end]), pmpo_sind[end]])[1]
    # bsizes = [buffer_size(buf[i]) for i in 1:2]
    for i in 1:2
        allocate_buffer!(buf[i])
    end
    #################################################
    #
    #             Left Enviroment
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
    step = ini_l ? 1 : 0
    for i in 1:step
        L, en0 = lproduct(L)
    end
    L, _ = anderson_accelerate(L, lproduct; tol, outputlevel, m=env_dim, gc)
    # mpo.L0 = lproduct(L, rep=replace)[1]
    replaceind!(L, dag(lind), rind)
    replaceind!(L, prime(lind), dag(prime(rind)))
    mpo.L0 = L
    #################################################
    #
    #             Right Enviroment
    #
    ################################################

    site_psi = isiteinds(psi_right)
    lind = setdiff(uniqueinds(psi_right[1], psi_right[2]), site_psi)[1]
    rind = setdiff(uniqueinds(psi_right[end], psi_right[end-1]), site_psi)[1]
    b = uniqueind(S0, psi_right[end]) #right indices
    deten = delta(dag(b), prime(b))
    link_id = commonind(mpo.R0, mpo.H[end])
    ten_zero = ITensor(dag(link_id))
    ten_zero[link_id=>end] = 1
    function rproduct(L; rep=true)
        en_density = local_energy(L, S0, ten_zero, deten) / Nuc
        #substract energy from current hamiltonian
        energyMPOSubtraction!(mpo, en_density)
        L = right_TMv(L, buf, lind, rind, mpo_lind, mpo_rind, psi_right, mpo, replace_inds=rep)
        energyMPOSubtraction!(mpo, -en_density)
        return L, en_density
    end

    L = mpo.R0
    step = ini_r ? 1 : 0
    for i in 1:step
        L, en0 = rproduct(L)
    end
    L, _ = anderson_accelerate(L, rproduct; tol, outputlevel, m=env_dim, gc)
    # mpo.R0 = rproduct(L, rep=replace)[1]
    replaceind!(L, dag(rind), lind)
    replaceind!(L, prime(rind), dag(prime(lind)))
    mpo.R0 = L
    for i in 1:2
        free_buffer!(buf[i])
    end
    return nothing
end
