include("grassmann.jl")
include("grad_env.jl")
include("grads.jl")
include("grad.jl")

function grad_opt(ipsi, mpo; nstep_max=20, observer = NoObserver(), gc_dim = 10000, buf = [DefaultBuffer(), DefaultBuffer()], env_dim = 5, kwargs...)
    psi = ipsi.psi
    S0 = ipsi.S0

    N = length(psi)
    vumps = vumps_canonical(N)

    psi, err = vumps_canonical_form(psi, S0, vumps)
    psi = vumps_initializeIMPO!(psi, mpo.H0, mpo, vumps, buf; S0, err=1e-12, ini_l=true, ini_r=true, gc_dim=10000, env_dim)

    eng_c, S0 = central_site_problem(psi, mpo, 1e-10, buf, lambda=S0)
    # res, psi, S0 = grad_descent!(psi, vumps, mpo, S0, buf; max_iter=grad_step)
    res, psi, S0 = cg_descent!(psi, vumps, mpo, S0, buf, observer; max_iter=nstep_max, kwargs...)

    psi, err = vumps_canonical_form(psi, S0, vumps)
    @printf "GC Finished, canonical error:%.2E\n" err
    

    ipsi.psi = psi
    ipsi.S0 = S0
    return ipsi
end
