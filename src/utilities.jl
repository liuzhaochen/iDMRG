mutable struct lanczo_cache
    Lv::ITensor
    LHv::ITensor
    function lanczo_cache()
        return new(ITensor(1.0), ITensor(1.0))
    end
end
function mpo_product(L, R, H, cache::lanczo_cache, c::ITensor, v)
    if cache.Lv == ITensor(1.0)
        cache.Lv = L * H * dag(c) * prime(c)
    end
    try
        # cache.LHv = contract!(cache.LHv, cache.Lv, H)
        cache.LHv = contract!(cache.LHv, cache.Lv, v)
    catch
        cache.LHv = cache.Lv * v
    end
    v1 = noprime(cache.LHv * R)
    return v1
end
function mpo_product(L, R, H, cache::lanczo_cache, v)
    if cache.Lv == ITensor(1.0)
        cache.Lv = L * H
    end
    try
        # cache.LHv = contract!(cache.LHv, cache.Lv, H)
        cache.LHv = contract!(cache.LHv, cache.Lv, v)
    catch
        cache.LHv = cache.Lv * v
    end
    v1 = noprime(cache.LHv * R)
    return v1
end
function mpo_product(L, R, v)
    #using in-place version
    v1 = noprime!((L * v) * R)
    return v1
end
function pseudo_id(lambda)
    #changte lambda into identity matrix
    if lambda == ITensor(1.0)
    else
        lambda = copy(lambda)
        lam_dim = size(lambda, 1)
        for i in 1:lam_dim
            val = lambda[i, i]
            lambda[i, i] = 1
        end
    end
    return lambda
end

function isproduct(psi::MPS)
    #check whether the psi has not links between unitcell
    ispro = false
    N = length(psi)
    for i in 1:N
        ind = inds(psi[i])
        if length(ind) < 3
            ispro = true
        end
    end
    return ispro
end
function circshift(psi::MPS, shift::Int)
    N = length(psi)
    tensor = [psi[i] for i in 1:N]
    circshift!(tensor, shift)
    return MPS(tensor)
end
function circshift(mpo::MPO, shift::Int)
    N = length(mpo)
    tensor = [mpo[i] for i in 1:N]
    circshift!(tensor, shift)
    return MPO(tensor)
end
function new_ind(id)
    id_new = Index(id.space)
    if dir(id_new) != dir(id)
        id_new = dag(id_new)
    end
    return id_new
end
function new_site_inds(swap_poi::Int, sites_odd)
    # phy = string(collect(tags(sites_odd[1]))[1])
    N = length(sites_odd)
    # withqn = hasqns(sites_odd[1]) ? true : false

    # sites_new = siteinds(phy, N, conserve_qns=withqn)
    sites_new = copy(sites_odd)
    for i in 1:N
        id = sites_odd[i]
        sites_new[i] = settags(new_ind(id), tags(id))
    end
    circshift!(sites_new, -swap_poi)
    return sites_new
end
function isiteinds(psi)
    #for open boundary psi
    #using a different method of siteinds
    idx = Index[]
    N = length(psi)
    for i in 1:N
        ids = inds(psi[i])
        for id in ids
            if hastags(id, "Site")
                push!(idx, noprime(id))
            end
        end
    end
    return unique(idx)
end
function filter_kwargs(kws, allowed)
    # clean_kwargs = Dict(k => v for (k, v) in kws if k in allowed)
    return (; (k => kws[k] for k in keys(kws) if k in allowed)...)
end
function add!(A, B, c)
    A .+= c .* B
end
function jd_correction(f, theta, u0, x0)
    # x = x0
    x = copy(x0)
    a = inner(u0, x)
    add!(x, u0, -a)
    hv = f(x)
    add!(hv, x, -theta)
    b = inner(u0, hv)
    add!(hv, u0, -b)
    return hv
end
function jacobi_davidson(f, x0;
    eigsolve_krylovdim=30,
    cg_krylovdim=5,
    eigsolve_maxiter=0,
    eigsolve_tol=1e-10,
    verbosity=0)

    energy = 0
    err = 0
    K = eigsolve_krylovdim
    tol = eigsolve_tol

    vec = Vector{ITensor}(undef, K + 1)
    Hvs = Vector{ITensor}(undef, K + 1)
    T = eltype(x0)
    A = zeros(T, K + 1, K + 1)


    vec[1] = copy(x0) / inner(x0, x0)
    Hv = f(x0)
    t0 = copy(x0)
    t0 .*= 0

    Hvs[1] = copy(Hv)
    A[1, 1] = real(inner(x0, Hv))
    theta = A[1, 1]
    X = copy(vec[1])
    r = Hv - theta * X
    err = norm(r)

    inf0 = nothing
    nit = 1#i + inf0.numops
    for i in 2:K+1
        if err < eigsolve_tol
            verbosity > 0 && @show i - 1, err, nit
            energy = theta
            break
        end
        alpha = inner(r, X)
        t0 .= -r
        add!(t0, X, alpha)
        t, inf0 = linsolve(x -> jd_correction(f, theta, X, x), -r, t0; atol=max(1e-12, err^2),
            krylovdim=cg_krylovdim, maxiter=1, ishermitian=true, verbosity=0)
        nit += inf0.numops
        # t = t0
        #update t0 by t in next iteration
        # t0 .= t

        for j in 1:i-1
            a = inner(vec[j], t)
            add!(t, vec[j], -a)
        end
        nrm = norm(t)
        if nrm < 1e-14
            break
        end
        t .*= 1 / nrm

        vec[i] = t
        Hvs[i] = f(t)
        nit += 1

        #build overlap matrix
        Ht = Hvs[i]
        for j in 1:i-1
            val = inner(vec[j], Ht)
            A[j, i] = val
            A[i, j] = conj(val)
        end
        A[i, i] = real(inner(t, Ht))


        eig, eigvecs = eigen(A[1:i, 1:i])
        theta = eig[1]
        eigvecs = eigvecs[:, 1]
        #build u matrix
        #vec[1] is not zero type 
        X .= eigvecs[1] .* vec[1]
        for j in 2:i
            add!(X, vec[j], eigvecs[j])
        end
        Hv .*= 0
        for j in 1:i
            add!(Hv, Hvs[j], eigvecs[j])
        end
        r .= Hv
        add!(r, X, -theta)
        err = norm(r)
    end
    return theta, X, err
end
