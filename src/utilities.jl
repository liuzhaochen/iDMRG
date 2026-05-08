
function mpo_product(L, R, Lv, LHv, H, v)
    #using in-place version
    # v1 = noprime!((L*v)*R)
    Lv = !isempty(commoninds(L, v)) ? contract!(Lv, L, v, 1.0, 0.0) : L*v
    LHv = contract!(LHv, Lv, H, 1.0, 0.0)
    v1 = noprime(LHv*R)
    return v1
end
function mpo_product(L, R, v)
    #using in-place version
    v1 = noprime!((L*v)*R)
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
function new_site_inds(swap_poi::Int,sites_odd)
    # phy = string(collect(tags(sites_odd[1]))[1])
    N = length(sites_odd)
    # withqn = hasqns(sites_odd[1]) ? true : false

    # sites_new = siteinds(phy, N, conserve_qns=withqn)
    sites_new = copy(sites_odd)
    for i in 1:N
        id =sites_odd[i] 
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
            # @show tags(id)
            phy = string(collect(tags(id))[2])
            if phy == "Site"
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
