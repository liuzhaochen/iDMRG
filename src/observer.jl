#iDMRG observer
abstract type Observer end
struct NoObserver <: Observer end
#check if calculation is done or saving the calculation during idmrg sweeps
checkdone!(o::Observer; kwargs...) = false
