
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
function new_site_inds(sites_odd)
    phy = string(collect(tags(sites_odd[1]))[1])
    N = length(sites_odd)
    withqn = hasqns(sites_odd[1]) ? true : false

    sites_new = siteinds(phy, N, conserve_qns=withqn)
    return sites_new
end
function isiteinds(psi)
    #for open boundary psi
    #using a different method of siteinds
    idx = []
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
function lambdamodule(l1, l2)
    return 0
    # size1 = size(l1, 1)
    # size2 = size(l2, 1)
    # largel1 = Float64[]
    # largel2 = Float64[]
    # for i = 1:size1
    #     push!(largel1, l1[i, i])
    # end
    # sort!(largel1)
    # for i = 1:size2
    #     push!(largel2, l2[i, i])
    # end
    # sort!(largel2)
    # ove = 0
    # for i in 1:min(size1, size2)
    #     ove += largel1[i] * largel2[i]
    # end
    # return ove
end
