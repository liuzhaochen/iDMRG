function projection_tangent(A::ITensor, W::ITensor, rind)
    #project W onto the tangent space of A
    # return W - A * A' * W 
    #indices of i-A-j
    #indices of W: -i-W-j-
    AdW = prime(A, rind) * (prime(dag(A), dag(rind)) * W)
    # add!(W, AdW, -1)
    W .+= -AdW
    # add!(W0, WdA, -1 / 2)
    return W
end
function projection_tangent!(A::MPS, W::MPS)
    N = length(A)
    sites = isiteinds(A)
    for i in 1:N
        rind = i != N ? commonind(A[i], A[i+1]) : setdiff(uniqueinds(A[i], A[i-1]), sites[i:i])[1]
        projection_tangent(A[i], W[i], rind)
    end
end
function polar_factor(X::ITensor, linds)
    U, S, V = svd(X, linds)
    S = pseudo_id(S)
    return U * S * V
end
function polar_factor!(X::MPS)
    N = length(X)
    x_polars = X
    sites = isiteinds(X)
    for i in 1:N
        linds = i != N ? uniqueinds(X[i], X[i+1]) : [sites[i], commoninds(X[i], X[i-1])]
        x_polars[i] .= polar_factor(X[i], linds)
    end
    return x_polars
end
function inner_imps(A::MPS, B::MPS)
    #using tensor wise product, as we view the problem with input A[1], A[2], A[3], not a whole MPS
    L = 0
    N = length(A)
    for i in 1:N
        L += (dag(A[i])*B[i])[]
    end
    return L
end

struct grassmann
    u::ITensor
    s::ITensor
    v::ITensor
    #w=u*s*v
end
function precondition(grad, precond)
    N = length(grad)
    grad1 = MPS(N)
    for i in 1:N
        grad1[i] = noprime(grad[i] * precond[i])
    end
    return grad1
end
function grassmanns(grad::MPS; precond=nothing)
    N = length(grad)
    grads = Vector{grassmann}(undef, N)
    sites = isiteinds(grad)
    for i in 1:N
        A = grad[i]
        if !isnothing(precond)
            A = noprime(A * precond[i])
        end
        if i != N
            inds = uniqueinds(grad[i], grad[i+1])
        else
            inds = [commonind(grad[i], grad[i-1]), sites[i]]
        end
        u, s, v = svd(A, inds)
        grads[i] = grassmann(u, s, v)
    end
    return grads
end
function St(fun, S, t)
    S0 = copy(S)
    dim = size(S, 1)
    for i in 1:dim
        val = S0[i, i]
        S0[i, i] = fun(val * t)
    end
    return S0
end
function OneMinusSt(fun, S, t)
    S0 = copy(S)
    dim = size(S, 1)
    for i in 1:dim
        val = S0[i, i]
        S0[i, i] = 1 - fun(val * t)
    end
    return S0
end
function St2(fun, S, t)
    S0 = copy(S)
    dim = size(S, 1)
    for i in 1:dim
        val = S0[i, i]
        S0[i, i] = val * fun(val * t)
    end
    return S0
end
function retract!(t, psi_l::MPS, grads::Vector{grassmann})
    #retract the psi_l
    N = length(psi_l)
    for i in 1:N
        u = grads[i].u
        s = grads[i].s
        v = grads[i].v

        a = commonind(s, u)
        b = commonind(s, v)
        c = uniqueind(v, s)

        sv = St(cos, s, t) * v
        # replaceind!(sv, a, b)
        b_v = commonind(v, s)
        ITensors.setinds!(sv, map(i -> i == a ? b_v : i, inds(sv)))
        v1sv = prime(dag(v), c) * sv
        usv = u * St(sin, s, t) * v
        #update psi_l
        psi_l[i] = prime(psi_l[i], c) * v1sv + usv
        # psi_l[i] = psi_l[i] + usv
    end
end
function parallel_transport(t, psi_l::MPS, grads::Vector{grassmann})
    #parallel transport vector grads along it self by t
    N = length(psi_l)
    grad_new = MPS(N)
    for i in 1:N
        u = grads[i].u
        s = grads[i].s
        v = grads[i].v

        a = commonind(s, u)
        b = commonind(s, v)
        c = uniqueind(v, s)

        sv = St2(sin, s, t) * v
        b_v = commonind(v, s)
        # ITensors.setinds!(sv, map(i -> i == a ? b_v : i, inds(sv)))
        replaceind!(sv, a, b)
        v1sv = prime(dag(v), c) * sv
        usv = u * St2(cos, s, t) * v
        #update psi_l
        grad_new[i] = -prime(psi_l[i], c) * v1sv + usv
    end
    return grad_new
end
function parallel_transport!(t, psi_l::MPS, H::Vector{grassmann}, grad::MPS)
    #parallel transport the grad vector at psi_l along H by step t
    N = length(psi_l)
    # grad_new = MPS(N)
    for i in 1:N
        u = H[i].u
        s = H[i].s
        v = H[i].v
        a = uniqueind(u, s)
        b = commonind(s, u)
        c = commonind(s, v)
        d = uniqueind(v, s)

        sv = psi_l[i] * dag(v) * dag(St(sin, s, t)) * dag(u)# * grad[i]
        su = u * OneMinusSt(cos, s, t)
        b_v = commonind(u, s)
        ITensors.setinds!(su, map(i -> i == c ? b_v : i, inds(su)))
        # replaceind!(su, c, dag(b))
        # sv += su * dag(u)
        #update psi_l
        grad[i] = grad[i] - (sv + su * dag(u)) * grad[i]
        # grad_new[i] = -prime(psi_l[i], c) * v1sv + usv
    end
    return grad
end
function parallel_transport(t, psi_l::MPS, H::Vector{grassmann}, grad::MPS)
    #parallel transport the grad vector at psi_l along H by step t
    N = length(psi_l)
    grad_new = MPS(N)
    for i in 1:N
        u = H[i].u
        s = H[i].s
        v = H[i].v
        a = uniqueind(u, s)
        b = commonind(s, u)
        c = commonind(s, v)
        d = uniqueind(v, s)

        sv = psi_l[i] * dag(v) * dag(St(sin, s, t)) * dag(u)# * grad[i]
        su = u * OneMinusSt(cos, s, t)
        b_v = commonind(u, s)
        ITensors.setinds!(su, map(i -> i == c ? b_v : i, inds(su)))
        # replaceind!(su, c, dag(b))
        # sv += su * dag(u)
        #update psi_l
        grad_new[i] = grad[i] - (sv + su * dag(u)) * grad[i]
        # grad_new[i] = -prime(psi_l[i], c) * v1sv + usv
    end
    return grad_new
end
