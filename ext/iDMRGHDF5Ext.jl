module iDMRGHDF5Ext

using HDF5: HDF5, attributes, create_group, open_group, read, write
using iDMRG: iMPS
using ITensorMPS: MPS
using ITensors: ITensor

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, M::iMPS)
    g = create_group(parent, name)
    attributes(g)["type"] = "iMPS"
    attributes(g)["version"] = 1
    write(g, "psi", M.psi)
    write(g, "S0", M.S0)
    return nothing
end
function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{iMPS})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "iMPS"
        error("HDF5 group or file does not contain iMPS data")
    end
    psi = read(g, "psi", MPS)
    S0 = read(g, "S0", ITensor)
    return iMPS(psi, S0)
end

end
