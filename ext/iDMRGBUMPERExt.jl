module iDMRGBUMPERExt


using Bumper
using iDMRG.ITensors
using iDMRG.ITensors.NDTensors
import iDMRG: AbstractBuffer, with_alloc_buffer, to_buffer, reset!,
    move_to_heap, checkpoint_buffer, buffer_restore!, buffer_size
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
    b.buf = Libc.malloc(Int(target))
    b.buf_len = target
    return b
end
function resize_buffer!(bsize, A::Allocator)
    Libc.free(A.buf.buf)
    A.buf.buf = Libc.malloc(Int(A.buf.bsize))
    A.buf.max_offset = UInt(0)
    #reset buffer to initial condition
    Bumper.reset_buffer!(buf.buf)
end
function reset!(buf::Allocator)
    resize!(buf.buf)
    Bumper.reset_buffer!(buf.buf)
    return buf
end
move_to_heap(A, buf::Allocator) = copy(A)

function buffer_size(bsize, buf::Allocator)
    return buf.buf.buf_len
end

end
