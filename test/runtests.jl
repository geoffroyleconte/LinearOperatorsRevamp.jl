using Arpack, Test, TestSetExtensions, LinearOperatorsRevamp, LinearAlgebra

include("test_aux.jl")

include("test_linop_allocs.jl")
include("test_linop.jl")
include("test_adjtrans.jl")
include("test_cat.jl")
