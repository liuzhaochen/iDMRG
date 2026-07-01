module iDMRGBUMPERExt


using Bumper
using iDMRG.ITensors
using iDMRG.ITensors.NDTensors
import iDMRG: AbstractBuffer, with_alloc_buffer, to_buffer, reset!,
    move_to_heap, buffer_size, resize_buffer!, free_buffer!, lanczo_cache_buffer
struct Allocator <: AbstractBuffer
    buf::Bumper.ResizeBuffer
    function Allocator(size=1048576)
        return new(Bumper.ResizeBuffer(size))
    end
end

with_alloc_buffer(f, buf::Allocator) = NDTensors.with_alloc_buffer(f, buf.buf)
to_buffer(A, buf::Allocator) = ITensors.to_buffer(A, buf.buf)
checkpoint_buffer(buf::Allocator) = Bumper.checkpoint_save(buf.buf)
buffer_restore!(cp::Bumper.ResizeBufferImpl.ResizeCheckpoint) = Bumper.checkpoint_restore!(cp)
@noinline function resize!(b::Bumper.ResizeBuffer)
    Libc.free(b.buf)
    target = b.max_offset * 11 ÷ 10
    b.buf = Libc.malloc(target)
    b.buf_len = target
    b.offset = UInt(0)
    b.max_offset = UInt(0)
    return b
end
function allocate_buffer!(A::Allocator)
    #using current max_offset to allocate a new buffer
    resize!(A.buf)
    return A
end
function free_buffer!(A::Allocator)
    Libc.free(A.buf.buf)
    A.buf.buf = Libc.malloc(2^20)
    A.buf.buf_len=UInt(2^20)
    return A
    #free buffer size but keep max_offset unchanged
end
function reset!(A::Allocator)
    resize!(A.buf)
    #this will reset overflow part of pointer
    Bumper.reset_buffer!(A.buf)
    return A
end
move_to_heap(A, buf::Allocator) = copy(A)

function buffer_size(bsize, buf::Allocator)
    return buf.buf.buf_len
end
function lanczo_cache_buffer(L, R, psi, A::Allocator)
    return ITensors.lanczos_permute(L, R, psi, A.buf)
end
function lanczo_cache_buffer(L, R, H0, H1, psi, A::Allocator)
    # return ITensors.lanczos_permute(L, R, psi, A.buf)
    L1 = to_buffer(L, A)
    R1 = to_buffer(R, A)
    H0_b = to_buffer(H0, A)
    H1_b = to_buffer(H1, A)
    psi = to_buffer(psi, A)
    return L1, R1, H0_b, H1_b, psi
end

end
