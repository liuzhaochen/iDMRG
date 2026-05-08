#solving the eigenvalue of MPS transfer matrix
function mps_TMv(psi_left::MPS, lind, rind, sites, v::ITensor)
    Nsite = length(psi_left)
    for j in 1:Nsite
        A = psi_left[j]
        v = (v*A)*dag(prime(A,!sites[j]))
    end
    #replace indices back 
    replaceind!(v, rind, dag(lind))
    replaceind!(v, dag(prime(rind)), prime(lind))
    return v
end
function mps_TM_spectrum(psi_left::MPS;nval=10)
    #the left transfer matrix eigenvalue
    site_psi = isiteinds(psi_left)
    lind = setdiff(uniqueinds(psi_left[1], psi_left[2]), site_psi)[1]
    rind = setdiff(uniqueinds(psi_left[end], psi_left[end-1]), site_psi)[1]

    v0 = random_itensor(dag(lind),prime(lind))
    #solve the spectrum
    product(x) = mps_TMv(psi_left, lind, rind, site_psi, x)
    vals, vecs,info = eigsolve(
        x -> product(x),
        v0,
        nval,
        :LM;
        tol=1e-14,
        krylovdim=20,
        maxiter=20,
        verbosity=0
    )
    @show info
    return vals
end
