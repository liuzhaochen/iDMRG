function vumps_grad(
    PH,
    C0::ITensor,
    vumps;
    write_when_maxdim_exceeds=nothing,
    write_path=tempdir(),
    kwargs...
)
    psi = vumps.psi_l
    C0 = deepcopy(C0)
    N = length(vumps.psi_l)
    err = 0.0
    energy = 0.0
    grad = MPS(N)
    precond = MPS(N)
    sites = isiteinds(psi)
    dx = -1
    A = vumps.psi_l
    hes = 0
    sw_time = @elapsed begin
        if !isnothing(write_when_maxdim_exceeds)
            if (maxlinkdim(psi) > write_when_maxdim_exceeds)
                PH = disk(PH; path=write_path)
            end
        end
        for b in N:-1:1
            PH = position!(PH, vumps.psi_l, b)
            poi_l = b + dx
            L = lproj(PH)
            R = rproj(PH)
            W = PH.H[b]

            g1 = noprime!(L * psi[b] * W * R)
            grad[b] = g1 # copy(g1)
            energy = (grad[b]*dag(vumps.psi_l[b]))[]


            lind = uniqueind(vumps.psi_l[b], C0)
            det = delta(dag(lind), prime(lind))
            rho = det * vumps.psi_l[b] * C0
            # no = rho * dag(prime(vumps.psi_l[b], !sites[b]))
            g2 = noprime(rho)
            add!(grad[b], g2, -energy)

            ng = real((dag(grad[b])*grad[b])[])
            err += ng


            #pseudo-inv of C.C^dag
            # u, s, v = svd(C0, lind)
            # rho0 = u * pseudo_inv(s, tol=max(eps()^(3 / 4), ng)) * v
            # precond[b] = rho0

            #calculate <g|H|g>
            # hes += (L*grad[b]*W*dag(prime(grad[b]))*R)[]
            # hes -= energy * (det*grad[b]*dag(prime(grad[b], !sites[b]))*C0)[]
            # hes -= 2real(inner(grad[b], g1) * inner(g2, grad[b]))
            # hes += 2energy * abs(inner(grad[b], g2))^2
            A = vumps.psi_l[b]
            C0 = C0 * dag(prime(A, !sites[b])) * A
        end
    end
    dt = 0 #err / hes
    return (real(energy), grad, precond, sqrt(err), dt)
end
function pseudo_inv(S; tol=1e-7)
    lam_dim = size(S, 1)
    s_max = maximum(S)
    delta2 = (tol * s_max)^2
    for i in 1:lam_dim
        val = S[i, i]
        S[i, i] = 1 / (val + delta2)
    end
    return S
end
