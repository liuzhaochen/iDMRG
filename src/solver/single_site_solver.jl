
function buffer_product(phi, L, R, buf)
    L1 = with_alloc_buffer(buf[2]) do
        L * phi
    end
    Hx = noprime!(L1 * R)
    reset!(buf[2])
    return Hx
end
function fp_eq(phi, L, R, buf, solver_para)
    if typeof(buf[1]) == HeapBuffer
        vals, vecs, info = eigsolve(
            x -> noprime!(L * x * R),
            phi,
            1,
            :SR;
            ishermitian=true,
            tol=solver_para.tol,
            krylovdim=solver_para.krylovdim,
            maxiter=solver_para.maxiter,
            verbosity=solver_para.verbosity,
            eager=true,
        )
        return vals[1], vecs[1], info.normres[1]
    else
        return vals, vecs, err = lanczos(
            buffer_product,
            phi,
            L,
            R,
            buf,
            tol=solver_para.tol,
            krylovdim=solver_para.krylovdim,
            maxiter=solver_para.maxiter,
            verbosity=solver_para.verbosity,
        )
    end
end
function single_site_eig(phi::ITensor, L::ITensor, R::ITensor, H0::ITensor, buf; solver_para)
    # L = lproj(PH)
    # R = rproj(PH)
    # H0 = PH.H[PH.lpos+1]
    LH = L * H0
    #here, we create a new buffer and move LH, R into it
    #and then free LH
    return fp_eq(phi, LH, R, buf, solver_para)
end
function vumps_bond_left_solve(i::Int, PH, psi_l::MPS, C, buf; solver_para)
    L = lproj(PH)
    R = rproj(PH)
    H = PH.H[i]
    phi = psi_l[i]
    #update L 
    L = (((L * phi) * H) * prime(dag(phi)))
    return fp_eq(C, L, R, buf, solver_para)
    # l_cache = lanczo_cache()
    # vals, vecs, info = eigsolve(
    #     x -> bond_product(L, R, l_cache, x),
    #     C,
    #     1,
    #     :SR;
    #     ishermitian=true,
    #     tol=solver_para.tol,
    #     krylovdim=solver_para.krylovdim,
    #     maxiter=solver_para.maxiter,
    #     verbosity=solver_para.verbosity,
    #     eager=true,
    # )
    # l_cache = nothing
    # L = nothing
    # return vals[1], vecs[1]
end
function vumps_bond_right_solve(i::Int, PH, psi_r::MPS, C, buf; solver_para)
    L = lproj(PH)
    R = rproj(PH)
    H = PH.H[i]
    phi = psi_r[i]
    #update R 
    R = (((R * phi) * H) * prime(dag(phi)))
    return fp_eq(C, R, L, buf, solver_para)
    # l_cache = lanczo_cache()
    # vals, vecs, info = eigsolve(
    #     x -> bond_product(L, R, l_cache, x),
    #     C,
    #     1,
    #     :SR;
    #     ishermitian=true,
    #     tol=solver_para.tol,
    #     krylovdim=solver_para.krylovdim,
    #     maxiter=solver_para.maxiter,
    #     verbosity=solver_para.verbosity,
    #     eager=true,
    # )
    # l_cache = nothing
    # R = nothing
    # return vals[1], vecs[1]
end
