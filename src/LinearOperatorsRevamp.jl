module LinearOperatorsRevamp

using FastClosures, LinearAlgebra, Printf, SparseArrays

# Basic defitions
include("abstract.jl")
include("constructors.jl")
include("PreallocatedLinearOperators.jl")

# Operations
include("operations.jl") # This first
include("adjtrans.jl")

include("special-operators.jl")

end # module
