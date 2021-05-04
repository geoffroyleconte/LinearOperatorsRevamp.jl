module LinearOperatorsRevamp

using FastClosures, LinearAlgebra, Printf, SparseArrays

# Basic defitions
include("abstract.jl")
include("constructors.jl")

# Operations
include("operations.jl") # This first

end # module
