function buffer_product(phi, L, R, H0, H1, buf)
    L1 = with_alloc_buffer(buf[2]) do
        L * phi * H0 * H1
    end
    Hx = noprime!(L1 * R)
    reset!(buf[2])
    return Hx
end

function two_site_eig(phi::ITensor, L::ITensor, R::ITensor, H0::ITensor, H1::ITensor, buf; solver_para)
    if typeof(buf[1]) == HeapBuffer
        vals, vecs, info = eigsolve(
            x -> noprime!(L * x * H0 * H1 * R),
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
        return vals, vecs, err = lanczos_2s(
            buffer_product,
            phi,
            L,
            R,
            H0,
            H1,
            buf,
            tol=solver_para.tol,
            krylovdim=solver_para.krylovdim,
            maxiter=solver_para.maxiter,
            verbosity=solver_para.verbosity,
        )
    end
end
