mutable struct eig_para
    tol::Float64
    krylovdim::Int
    maxiter::Int
    verbosity::Int
    function eig_para(eigsolve_tol, eigsolve_krylovdim, eigsolve_maxiter)
        return new(1.0eigsolve_tol, eigsolve_krylovdim, eigsolve_maxiter, 0)
    end
end
include("single_site_solver.jl")
include("two_site_solver.jl")
