mutable struct Lanczos
    K::Int #maximum rank
    poi::Int #current level
    V::Vector{ITensor}
    as::Vector{Float64}
    bs::Vector{Float64}
    r::ITensor
end
Base.length(A::Lanczos) = A.poi
normres(A::Lanczos) = A.bs[A.poi]
function append!(lan::Lanczos, x::ITensor)
    i = lan.poi + 1
    lan.V[i] = x
    lan.poi += 1
end
function apply!(f!, x0::ITensor)
    # x_temp = A*x0
    return f!(x0)
end
function initialize(f!, x0::ITensor, K::Int)
    #aways assume norm(x0) = 1 as start point
    vec = Vector{ITensor}(undef, K + 1)
    alpha = zeros(K + 1)
    beta = zeros(K + 1)
    x0 = copy(x0) / norm(x0)
    lan = Lanczos(K, 0, vec, alpha, beta, ITensor(1.0))
    w = apply!(f!, x0)
    append!(lan, x0)

    α = inner(x0, w)
    lan.r = w
    βold = norm(lan.r)
    # lan.r = copy!(lan.r, lan.x_tmp)
    # lan.r = add!(lan.r, x0, -a0)
    lan.r .+= (-α) .* x0
    β = norm(lan.r)
    while eps(one(β)) < β < 1 / sqrt(2) * βold
        βold = β
        dα = inner(x0, lan.r)
        α += dα
        lan.r .+= (-dα) .* x0  # should we use real(dα) here?
        β = norm(lan.r)
    end

    lan.as[1] = real(α)
    lan.bs[1] = β
    return lan
end
function orthogonalize(v, q)
    # nold = norm(v)
    s = inner(q, v)
    # v = add!(v, q, -s)
    v .+= -s .* q
    nnew = norm(v)
    # while eps(one(nnew)) < nnew < 1 / sqrt(2) * nold
    #     nold = nnew
    #     ds = inner(q, v)
    #     # v = add!(v, q, -ds)
    #     v .+= -ds .* q
    #     s += ds
    #     nnew = norm(v)
    # end
    return (v, s)
end
function expand!(f!, lan::Lanczos)
    #expand the lanzcos space
    #reuse the ram as much as possible
    beta = normres(lan)
    if isnan(beta) || beta < 1e-12
        @show lan.bs
    end
    lan.r .*= 1 / beta
    # rescale!(lan.r, 1 / beta)
    append!(lan, lan.r) #this will make a copy if lan.V[end] is undef
    # w = A*v
    w = apply!(f!, lan.r)
    # w = add!(w, lan.V[lan.poi-1], -beta)
    w .+= -beta .* lan.V[lan.poi-1]


    v = lan.V[lan.poi]
    w, α = orthogonalize(w, v)
    ab2 = abs2(α) + abs2(beta)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < 1 / sqrt(2) * nold
        nold = β
        s = zero(α)
        for i in 1:lan.poi
            q = lan.V[i]
            w, s = orthogonalize(w, q)
        end
        α += s
        β = norm(w)
    end
    lan.r = w
    # lan.r = copy!(lan.r, w)
    lan.as[lan.poi] = real(α)
    lan.bs[lan.poi] = β
    return lan
end
function givens_rotation(a::Float64, b::Float64)
    if b == 0.0
        return 1.0, 0.0, a
    elseif abs(b) > abs(a)
        τ = -a / b
        s = 1.0 / sqrt(1.0 + τ^2)
        c = s * τ
        ρ = b / s
    else
        τ = -b / a
        c = 1.0 / sqrt(1.0 + τ^2)
        s = c * τ
        ρ = a / c
    end
    return c, s, ρ
end

function apply_givens_rows!(T::Matrix{Float64}, i1::Int, i2::Int, c::Float64, s::Float64)
    n = size(T, 2)
    for j in 1:n
        t1 = T[i1, j]
        t2 = T[i2, j]
        T[i1, j] = c * t1 - s * t2
        T[i2, j] = s * t1 + c * t2
    end
end

function apply_givens_cols!(T::Matrix{Float64}, j1::Int, j2::Int, c::Float64, s::Float64)
    m = size(T, 1)
    for i in 1:m
        t1 = T[i, j1]
        t2 = T[i, j2]
        T[i, j1] = c * t1 - s * t2
        T[i, j2] = s * t1 + c * t2
    end
end
function apply_givens_V!(lan::Lanczos, i1::Int, i2::Int, c::Float64, s::Float64)
    lan.r .= lan.V[i1]
    # lan.x_tmp = copy!(lan.x_tmp, lan.V[i1])
    lan.V[i1] .= (c .* lan.V[i1]) .+ (-s .* lan.V[i2])
    lan.V[i2] .= (c .* lan.V[i2]) .+ (s .* lan.r)
end
function restart!(lan::Lanczos, D, f_vals, keep::Int)
    poi = lan.poi
    p = poi - keep
    shifts = D[keep+1:end]

    # 1. 将当前残差向量归一化，放入 V[poi+1]
    β_old = lan.bs[poi]
    if isassigned(lan.V, poi + 1)
        lan.V[poi+1] .= lan.r
        # lan.V[poi+1] = copy!(lan.V[poi+1], lan.r)
    else
        lan.V[poi+1] = copy(lan.r)
    end
    lan.V[poi+1] .*= 1.0 / β_old
    # rescale!(lan.V[poi+1], 1.0 / β_old)

    # 2. 构造扩展三对角矩阵 T (poi+1)×(poi+1)，仅由 α, β 构成
    T = zeros(Float64, poi + 1, poi + 1)
    for i in 1:poi
        T[i, i] = lan.as[i]
    end
    for i in 1:poi-1
        T[i, i+1] = lan.bs[i]
        T[i+1, i] = lan.bs[i]
    end
    T[poi, poi+1] = β_old
    T[poi+1, poi] = β_old

    # 3. p 步隐式 QR 迭代（双位移优化）
    r = 1
    while r <= p
        if r < p
            # ========== 双隐式位移 ==========
            μ1 = shifts[r]
            μ2 = shifts[r+1]

            # 构造第一个 Givens 旋转，基于 2×2 前导子矩阵
            x1 = (T[1, 1] - μ1) * (T[1, 1] - μ2) + T[2, 1]^2
            x2 = T[2, 1] * (T[1, 1] + T[2, 2] - μ1 - μ2)

            if poi >= 3
                x3 = T[2, 1] * T[3, 2]
                # 第一个旋转消去 x2（和 x3 一起），这里简化：仅用 x1, x2 确定 c, s
                c, s, ρ = givens_rotation(x1, x2)
            else
                c, s, ρ = givens_rotation(x1, x2)
            end

            # 作用到 T 的第 1,2 行和列
            for j in 1:poi+1
                t1 = T[1, j]
                t2 = T[2, j]
                T[1, j] = c * t1 - s * t2
                T[2, j] = s * t1 + c * t2
            end
            for i in 1:poi+1
                t1 = T[i, 1]
                t2 = T[i, 2]
                T[i, 1] = c * t1 - s * t2
                T[i, 2] = s * t1 + c * t2
            end

            # 作用到 V
            apply_givens_V!(lan, 1, 2, c, s)

            # Bulge chase: 追逐填充元，从 i=2 到 poi
            for i in 2:poi
                x1 = T[i, i-1]
                x2 = T[i+1, i-1]
                c, s, ρ = givens_rotation(x1, x2)

                # 更新行 i 和 i+1
                for j in i-1:poi+1
                    t1 = T[i, j]
                    t2 = T[i+1, j]
                    T[i, j] = c * t1 - s * t2
                    T[i+1, j] = s * t1 + c * t2
                end
                # 更新列 i 和 i+1
                for j in 1:poi+1
                    t1 = T[j, i]
                    t2 = T[j, i+1]
                    T[j, i] = c * t1 - s * t2
                    T[j, i+1] = s * t1 + c * t2
                end

                apply_givens_V!(lan, i, i + 1, c, s)
            end

            r += 2
        else
            # ========== 单个位移（最后剩下的）==========
            μ = shifts[r]

            # 第一个 Givens 旋转在 (1,2)，消去 T[2,1] 并引入位移
            x1 = T[1, 1] - μ
            x2 = T[2, 1]
            c, s, ρ = givens_rotation(x1, x2)

            for j in 1:poi+1
                t1 = T[1, j]
                t2 = T[2, j]
                T[1, j] = c * t1 - s * t2
                T[2, j] = s * t1 + c * t2
            end
            for i in 1:poi+1
                t1 = T[i, 1]
                t2 = T[i, 2]
                T[i, 1] = c * t1 - s * t2
                T[i, 2] = s * t1 + c * t2
            end

            apply_givens_V!(lan, 1, 2, c, s)

            # Bulge chase
            for i in 2:poi
                x1 = T[i, i-1]
                x2 = T[i+1, i-1]
                c, s, ρ = givens_rotation(x1, x2)

                for j in i-1:poi+1
                    t1 = T[i, j]
                    t2 = T[i+1, j]
                    T[i, j] = c * t1 - s * t2
                    T[i+1, j] = s * t1 + c * t2
                end
                for j in 1:poi+1
                    t1 = T[j, i]
                    t2 = T[j, i+1]
                    T[j, i] = c * t1 - s * t2
                    T[j, i+1] = s * t1 + c * t2
                end

                apply_givens_V!(lan, i, i + 1, c, s)
            end

            r += 1
        end
    end

    # 4. 提取新的 Lanczos 分解
    for j in 1:keep
        lan.as[j] = T[j, j]
        lan.bs[j] = T[j+1, j]
    end

    # 5. 更新残差向量
    lan.r .= lan.V[keep+1]
    lan.r .*= T[keep+1, keep]

    lan.poi = keep
    return lan
end
function lanczos(f, x0::ITensor, LH::ITensor, R::ITensor, buf; tol=1e-10, maxiter=100, verbosity=0, krylovdim=32)
    eng = 0
    err = 0
    x0_b = to_buffer(x0, buf[1])
    for i in 1:maxiter
        L_b = to_buffer(LH, buf[1])
        R_b = to_buffer(R, buf[1])
        eng, x0, err = with_alloc_buffer(buf[1]) do
            lanczos_onestep(x -> f(x, L_b, R_b, buf), x0_b; tol, verbosity, krylovdim)
        end
        x0 = with_alloc_buffer(buf[2]) do
            #move x0(buf1) to buf2
            copy(x0)
        end
        if err < tol || i == maxiter
            x0 = move_to_heap(x0, buf[1])
            break
        end
        reset!(buf[1])
        x0_b = to_buffer(x0, buf[1])
    end
    reset!(buf[1])
    reset!(buf[2])
    return eng, x0, err
end
function lanczos_onestep(f, x0::ITensor; tol=1e-10, verbosity=0, krylovdim=32)
    lan = initialize(f, x0, krylovdim)
    count = 0
    numops = 1
    D = nothing
    U = nothing
    f_vals = nothing
    while true
        β = normres(lan)
        poi = length(lan)
        if poi > 0
            #diagonalize
            #this should be a symmetric vector as such the eigenvalue is already sorted
            T = SymTridiagonal(lan.as[1:poi], lan.bs[1:poi-1])
            D, U = eigen(T)
            f_vals = abs.(β * U[poi, :])
            theta = D[1]
            err = f_vals[1]
            if err < tol || poi == lan.K #count == maxiter
                #construct result and return
                lan.r .= 0
                for i in 1:poi
                    lan.r .+= U[i, 1] .* lan.V[i]
                    # add!(lan.x_tmp, lan.V[i], U[i, 1])
                end
                if verbosity > 0
                    @show poi, numops, err
                end
                return theta, lan.r, err
            end
        end
        if poi < lan.K
            expand!(f, lan)
            numops += 1
            # else
            # count += 1
            #     keep = Int(div(3 * lan.K, 5)) #we only need one eigenvalue
            #     restart!(lan, D, f_vals, keep)
        end
    end
end

