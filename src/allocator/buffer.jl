abstract type AbstractBuffer end
struct heapcp end
function with_alloc_buffer(f, buf::AbstractBuffer)
end

function to_buffer(A, buf::AbstractBuffer)
end

function reset!(buf::AbstractBuffer)
end

function checkpoint_buffer(buf::AbstractBuffer)
end

function resize_buffer!(bsize, buf::AbstractBuffer)
end

function buffer_size(buf::AbstractBuffer)
end
function buffer_restore!(cp)
end

struct HeapBuffer <: AbstractBuffer end

with_alloc_buffer(f, ::HeapBuffer) = f()
to_buffer(A, ::HeapBuffer) = A
reset!(::HeapBuffer) = nothing
move_to_heap(A, ::HeapBuffer) = A
checkpoint_buffer(::HeapBuffer) = heapcp()
buffer_restore!(cp::heapcp) = nothing
function DefaultBuffer(size=1048576)
    ext = Base.get_extension(@__MODULE__, :iDMRGBUMPERExt)
    if ext !== nothing
        return ext.Allocator(size)
    else
        HeapBuffer()
    end
end

function resize_buffer!(bsize, buf::HeapBuffer)
end
function buffer_size(buf::HeapBuffer)
    return UInt(0)
end
