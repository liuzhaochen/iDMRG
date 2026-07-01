const _using_buffer = Ref(false)
function enable_buffer()
    _using_buffer[] = true
    @printf "buffer: %s\n" using_buffer()
    return _using_buffer[]
end
function disable_buffer()
    _using_buffer[] = false
    @printf "buffer: %s\n" using_buffer()
    return _using_buffer[]
end
using_buffer() = _using_buffer[]
##################################
#
#
abstract type AbstractBuffer end
struct heapcp end
function with_alloc_buffer(f, buf::AbstractBuffer)
end

function to_buffer(A, buf::AbstractBuffer)
end

function reset!(buf::AbstractBuffer)
end

function resize_buffer!(bsize, buf::AbstractBuffer)
end

function buffer_size(buf::AbstractBuffer)
end

function free_buffer!(buf::AbstractBuffer)
end
function allocate_buffer!(buf::AbstractBuffer)
end
function lanczo_cache_buffer(L, R, psi, buff::AbstractBuffer)
    return L, R, psi
end

function lanczo_cache_buffer(L, R, H0, H1, psi, buff::AbstractBuffer)
    return L, R, H0, H1, psi
end
####################################################
#
#
struct HeapBuffer <: AbstractBuffer end

with_alloc_buffer(f, ::HeapBuffer) = f()
to_buffer(A, ::HeapBuffer) = A
reset!(::HeapBuffer) = nothing
move_to_heap(A, ::HeapBuffer) = A
checkpoint_buffer(::HeapBuffer) = heapcp()
buffer_restore!(cp::heapcp) = nothing

function resize_buffer!(bsize, buf::HeapBuffer)
end
function buffer_size(buf::HeapBuffer)
    return UInt(0)
end
function free_buffer!(buf::HeapBuffer)
end
function allocate_buffer!(buf::HeapBuffer)
end
function lanczo_cache_buffer(L, R, psi, buff::HeapBuffer)
    return L, R, psi
end
function lanczo_cache_buffer(L, R, H0, H1, psi, buff::HeapBuffer)
    return L, R, H0, H1, psi
end
############################################################
function DefaultBuffer(size=1048576)
    ext = Base.get_extension(@__MODULE__, :iDMRGBUMPERExt)
    if using_buffer()
        return ext.Allocator(size)
    else
        HeapBuffer()
    end
end
