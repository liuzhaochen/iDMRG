function update_vumps!(vumps, S0, mpo, buf; err=1e-8, ini=false)
    mpo.niter = 1
    mpo.lpos = 0
    mpo.rpos = length(mpo) + 1

    psi_left = vumps.psi_l
    grad_mpo_env!(psi_left, S0, vumps.C[end], mpo, buf; ini, tol=err)
    mpo.LR = Vector{ITensor}(undef, length(mpo))
    return nothing
end
function env_eng(vumps, mpo)
    lind = commonind(vumps.psi_l[1], mpo.L0)
    rind = commonind(vumps.psi_l[end], mpo.R0)
    lindo, rindo = iDMRG.mpo_env_linkinds(vumps.psi_l, mpo)

    R0 = copy(mpo.R0)
    replaceind!(R0, dag(rind), lind)
    replaceind!(R0, dag(prime(rind)), dag(prime(lind)))
    replaceind!(R0, rindo, dag(lindo))
    eng = (mpo.L0*R0)[]
    return eng
end
function left_eig(psi_l, x0, C0)
    #solve the right fixed point of psi_l
    # @show norm(C)
    site = iDMRG.isiteinds(psi_l)
    N = length(psi_l)
    lind = setdiff(uniqueinds(psi_l[1], psi_l[2]), site)[1]
    rind = commonind(psi_l[end], C0)
    function tm(psi_l, x)
        for i in N:-1:1
            A = psi_l[i]
            x = x * dag(prime(A, !site)) * A
            if length(inds(x)) > 2
                error(inds(x))
            end
        end
        #replace inds
        replaceind!(x, lind, dag(rind))
        replaceind!(x, dag(prime(lind)), prime(rind))
        return x
    end
    eig, vec = eigsolve(x -> tm(psi_l, x), x0, 1, :LM; ishermitian=true)
    x = vec[1]
    #need to normalize this
    x = x / tr(x)
    return x
end
function grad_one_step(psi, vumps, mpo, C0;)
    N = length(psi)
    C0 = left_eig(vumps.psi_l, C0, vumps.C[end])

    #need to estimate environment energy
    eng_c = env_eng(vumps, mpo)
    energyMPOSubtraction!(mpo, eng_c / N)
    eng, grad, precond, ng, dt = vumps_grad(mpo, C0, vumps)
    energyMPOSubtraction!(mpo, -eng_c / N)
    return grad, precond, eng / N, ng, C0, dt
end
function recover_S(C, vumps, S0, rind)
    #perform SVD for C matrix
    lind = commonind(C, vumps.psi_l[end])
    u, s, v = svd(C, lind; leftdir=ITensors.Out, rightdir=ITensors.Out)
    s = sqrtS(s)
    c_new = u * s
    c_new = c_new / norm(c_new)
    # rind0 = uniqueind(c_new, vumps.psi_l[end])
    # replaceind!(c_new, rind0, rind)
    lq = uniqueind(S0, c_new)
    S0 = copy(c_new)
    replaceind!(S0, lind, lq)
    return c_new, S0
end
function grad_descent!(psi, vumps, mpo, S0, buf; dt=1e-1, max_iter=500, mu=0.0, reset=5)
    # initializeIMPO!(psi, mpo.H0, mpo, vumps, buf)
    update_vumps!(vumps, S0, mpo, buf, ini=true)
    update_vumps!(vumps, S0, mpo, buf)
    C0 = vumps.C[end]
    rind = uniqueind(C0, vumps.psi_l[end])
    l = uniqueind(C0, vumps.psi_l[end])
    C = C0 * dag(prime(C0, !l))


    N = length(vumps.psi_l)
    res = Float64[]
    eng_prev = 1
    success_count = 0
    velocity = nothing
    for k in 1:max_iter
        sw_time = @elapsed begin
            grad, precond, eng, ng, C, dt0 = grad_one_step(psi, vumps, mpo, C)
            ng = real(ng)
            #apply precondition to gradient
            # Pgrad = precondition(grad, precond)
            Pgrad = grad
            if eng < eng_prev
                success_count += 1
                if success_count >= 3
                    dt *= 1.5
                    success_count = 0
                end
            else
                dt *= 0.5
                success_count = 0
            end
            eng_prev = eng
            # H = grassmanns(Pgrad)
            # retract!(-dt, vumps.psi_l, H)
            for i in 1:N
                add!(vumps.psi_l[i], Pgrad[i], -dt)
            end
            polar_factor!(vumps.psi_l)
            update_vumps!(vumps, S0, mpo, buf)
        end
        @printf "GS step: %i dt=%.1E grad=%.1E Eng=%s time=%.3f\n" k dt ng eng sw_time
        flush(stdout)
        if (dt <= 1e-8 || ng < 1e-8)
            break
        end
        eng_prev = eng
    end
    c_new, S0 = recover_S(C, vumps, S0, rind)
    psi = deepcopy(vumps.psi_l)
    psi[end] = psi[end] * c_new
    return res, psi, S0
end
function sqrtS(lambda)
    #changte lambda into identity matrix
    if lambda == ITensor(1.0)
    else
        lambda = copy(lambda)
        lam_dim = size(lambda, 1)
        for i in 1:lam_dim
            val = lambda[i, i]
            lambda[i, i] = sqrt(val)
        end
    end
    return lambda
end
function cg_descent!(psi, vumps, mpo, S0, buf, observer; dt=1.0, max_iter=500, tfac=2, grid=6, isgrid=false,
    isBB=true, kwargs...)
    update_vumps!(vumps, S0, mpo, buf, ini=true)
    update_vumps!(vumps, S0, mpo, buf)
    C0 = vumps.C[end]
    rind = uniqueind(C0, vumps.psi_l[end])
    l = uniqueind(C0, vumps.psi_l[end])
    C = C0 * dag(prime(C0, !l))
    N = length(vumps.psi_l)

    res = Float64[]
    eng_prev = 1
    success_count = 0
    velocity = nothing
    grad_prev = nothing
    Pgrad_prev = nothing
    ng_prev = 1
    mu = 0
    mu_1 = 1
    bb_1 = dt
    bb_2 = 1
    gd = -1
    eng_his = Float64[]
    grad, precond, eng, ng, C = grad_one_step(psi, vumps, mpo, C)
    push!(eng_his, eng)
    for k in 1:max_iter
        sw_time = @elapsed begin
            Pgrad = grad
            if k == 1
                velocity = deepcopy(Pgrad)
                for i in 1:N
                    velocity[i] .*= -1
                end
            else
                # project the velocity onto the tangent space
                projection_tangent!(vumps.psi_l, velocity)
                projection_tangent!(vumps.psi_l, Pgrad_prev)
                # PolakRibiere
                gPg = inner_imps(grad, Pgrad_prev)
                mu_0 = real(ng^2 - gPg)

                # gd = real(inner_imps(velocity, grad))
                # sy = gd - inner_imps(velocity, Pgrad_prev)
                # bb_1 = real(inner_imps(velocity, velocity) / sy * dt)
                # bb_2 = real(sy / mu_0 * dt)

                mu = min(5.0, mu_0 / mu_1)
                dg = real(inner_imps(velocity, grad))
                # kappa = abs(gPg / sqrt(ng * mu_1))
                if dg * mu - ng^2 > 0 || mod(k, 10) == 0
                    mu = 0
                end
                for i in 1:N
                    velocity[i] .*= mu
                    velocity[i] .+= -1.0 .* Pgrad[i]
                end
                dg = dg * mu - ng^2
            end
            #precond the grad
            Pgrad_prev = Pgrad
            mu_1 = real(inner_imps(grad, Pgrad_prev))
            ng_prev = ng
            # H = grassmanns(velocity)
            #parallel transport vector along seraching direction
            # while true          
            # if isgrid
            #     dt = line_search(psi, vumps, S0, C, mpo, H, dt, eng, buf)
            # elseif isBB
            #     dt = isodd(k) ? bb_1 : bb_2
            #     dt = clamp(dt, 1e-5, 2.0)
            #     dt = bbline_search(psi, vumps, S0, C, mpo, H, gd, dt, eng, buf, eng_his)
            # end
            # dt = line_search(psi, vumps, S0, C, mpo, H, dt, eng, buf)
            H = grassmanns(velocity)
            psi0 = copy(vumps.psi_l)
            grad, cond, eng, ng, C, dt = lineBT_search(psi, vumps, S0, C, mpo, H, gd,
                min(1 / abs(gd), 5.0), eng, buf, eng_his)

            velocity = parallel_transport!(dt, vumps.psi_l, H, velocity)
            Pgrad_prev = parallel_transport!(dt, vumps.psi_l, H, Pgrad_prev)
            push!(eng_his, eng)
            #update parameter 
            # retract!(dt, vumps.psi_l, H)
            #polar step, make sure the isometry
            # polar_factor!(vumps.psi_l)
            # update_vumps!(vumps, S0, mpo, buf;)
        end
        @printf "GS step: %i β=%.2f dt=%.1E grad=%.1E Eng=%s time=%.3f\n" k mu dt ng eng sw_time
        flush(stdout)

        c_new, S0 = recover_S(C, vumps, S0, rind)
        psi = copy(vumps.psi_l)
        psi[end] = psi[end] * c_new
        isdone = checkdone!(observer; eng_density=eng, step=(0, k), psi, S0) || k == max_iter
        if dt <= 1e-8 || ng < 1e-7 || isdone
            break
        end
    end
    return res, psi, S0
end
function line_search(psi, vumps, S0, C, mpo, H, dt, eng0, buf, tfac, grid)
    x0 = range(0, tfac, grid) * dt
    for j in 1:10
        dt0, eng_t = line_search(x0, psi, vumps, S0, C, mpo, H, dt, eng0, buf)
        if abs(dt0 - x0[end]) < 1e-10
            x0 = range(x0[end], 2x0[end], grid)
            eng = eng_t
        elseif dt0 == 0
            x0 = range(0, x0[2], grid)
        else
            dt = dt0
            break
        end
    end
    return dt
end
function line_search(x0, psi, vumps, S0, C, mpo, H, dt, eng0, buf)
    psi_l0 = deepcopy(vumps.psi_l)
    C0 = copy(C)
    #a simple line search
    function move(dx)
        #have to copy the initial points
        vumps.psi_l = deepcopy(psi_l0)
        retract!(dx, vumps.psi_l, H)
        polar_factor!(vumps.psi_l)
        update_vumps!(vumps, S0, mpo, buf)
        grad, pg, eng, ng, C = grad_one_step(psi, vumps, mpo, C0)
        return eng
    end
    #grid search
    nx = length(x0)
    eg = Float64[eng0]
    for i in 2:nx
        x = x0[i]
        push!(eg, move(x))
    end
    x_st = Float64[]
    eng_st = Float64[]
    for i in 1:length(x0)-2
        f1, f2, f3 = eg[i], eg[i+1], eg[i+2]
        x1 = x0[i]
        x2 = x0[i+1]
        x3 = x0[i+2]
        if (f2 <= f1 && f2 <= f3)
            denom = (x2 - x3) * f1 + (x3 - x1) * f2 + (x1 - x2) * f3
            x_try = 0.5 * ((x2^2 - x3^2) * f1 + (x3^2 - x1^2) * f2 + (x1^2 - x2^2) * f3) / denom
            #what is f(x_try)
            L1 = ((x_try - x2) * (x_try - x3)) / ((x1 - x2) * (x1 - x3))
            L2 = ((x_try - x1) * (x_try - x3)) / ((x2 - x1) * (x2 - x3))
            L3 = ((x_try - x1) * (x_try - x2)) / ((x3 - x1) * (x3 - x2))
            eng_try = f1 * L1 + f2 * L2 + f3 * L3

            push!(x_st, x_try)
            push!(eng_st, eng_try)
        end
    end
    vumps.psi_l = psi_l0
    if length(x_st) > 0
        i = argmin(eng_st)
        return x_st[i], eng_st[i]
    else
        i = argmin(eg)
        return x0[i], eg[i]
    end
end
function lineBT_search(psi, vumps, S0, C, mpo, H, gd, dt, eng0, buf, eng_his; max_iter=20, delta=1e-2)
    psi_l0 = deepcopy(vumps.psi_l)
    C0 = copy(C)
    #a simple line search
    gd = gd
    function move(dx)
        #have to copy the initial points
        vumps.psi_l = deepcopy(psi_l0)
        # H is svd of velocity, which is searching direction
        # here, transport direction to new point
        retract!(dx, vumps.psi_l, H)
        polar_factor!(vumps.psi_l)
        update_vumps!(vumps, S0, mpo, buf)
        grad, pg, eng, ng, C = grad_one_step(psi, vumps, mpo, C0)

        return grad, pg, eng, ng, C, dx
    end
    n_his = length(eng_his)
    e_ref = eng0   #sum(eng_his[max(1, n_his - m + 1):n_his])/m
    for i in 1:max_iter
        re = move(dt)
        eg = re[3]
        if eg < e_ref + 0delta * dt * gd || dt < 1e-5 || i == max_iter
            re = move(dt / 2)
            eg2 = re[3]
            f1, f2, f3 = eng0, eg2, eg
            x1 = 0
            x2 = dt / 2
            x3 = dt
            if (f2 <= f1 && f2 <= f3)
                denom = (x2 - x3) * f1 + (x3 - x1) * f2 + (x1 - x2) * f3
                x_try = 0.5 * ((x2^2 - x3^2) * f1 + (x3^2 - x1^2) * f2 + (x1^2 - x2^2) * f3) / denom
                return move(x_try)
            else
                return move(dt)
            end
        else
            dt *= 0.5
        end
    end
    return dt
end
