#disk MPO following : 
#https://github.com/ITensor/ITensorMPS.jl/blob/main/src/abstractprojmpo/diskprojmpo.jl of ITensorMPS package
mutable struct DiskIMPO <: AbstractProjMPO
    lpos::Int
    rpos::Int
    nsite::Int
    H::MPO
    L0::ITensor
    R0::ITensor
    LR::DiskVector{ITensor}
    Lcache::Union{ITensor, OneITensor}
    lposcache::Union{Int, Nothing}
    Rcache::Union{ITensor, OneITensor}
    rposcache::Union{Int, Nothing}
end

function ITensorMPS.set_nsite!(P::DiskIMPO, nsite)
    P.nsite = nsite
    return P
end
function ITensorMPS.disk(pm::iMPO; kwargs...)
    return DiskIMPO(
        pm.lpos,
        pm.rpos,
        pm.nsite,
        pm.H,
        pm.L0,
        pm.R0,
        disk(pm.LR; kwargs...),
        lproj(pm),
        pm.lpos,
        rproj(pm),
        pm.rpos
    )
end
ITensorMPS.disk(pm::DiskIMPO; kwargs...) = pm
function ITensorMPS.lproj(P::DiskIMPO)
    (P.lpos <= 0) && return P.L0
    if (P.lpos ≠ P.lposcache) || (P.lpos == 1)
        # Need to update the cache
        P.Lcache = P.LR[P.lpos]
        P.lposcache = P.lpos
    end
    return P.Lcache
end
function ITensorMPS.rproj(P::DiskIMPO)
    (P.rpos >= length(P) + 1) && return P.R0
    if (P.rpos ≠ P.rposcache) || (P.rpos == length(P))
        # Need to update the cache
        P.Rcache = P.LR[P.rpos]
        P.rposcache = P.rpos
    end
    return P.Rcache
end

function ITensorMPS.makeL!(P::DiskIMPO, psi::MPS, k::Int)
    L = ITensorMPS._makeL!(P, psi, k)
    if !isnothing(L)
        # Cache the result
        P.Lcache = L
        P.lposcache = P.lpos
    end
    return P
end

function ITensorMPS.makeR!(P::DiskIMPO, psi::MPS, k::Int)
    R = ITensorMPS._makeR!(P, psi, k)
    if !isnothing(R)
        # Cache the result
        P.Rcache = R
        P.rposcache = P.rpos
    end
    return P
end

function disk_cleanup!(P::iMPO)
end
function disk_cleanup!(P::DiskIMPO)
    rm(P.LR.pathname, recursive=true)
end
