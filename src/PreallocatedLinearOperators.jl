export PreallocatedLinearOperator

abstract type AbstractPreallocatedLinearOperator{T} <: AbstractLinearOperator{T} end

"""
Type to represent a linear operator with preallocation. Implicit modifications may
happen if used without care:
```
op = PreallocatedLinearOperator(rand(5, 5))
v  = rand(5)
x = op * v        # Uses internal storage and passes pointer to x
y = op * ones(5)  # Overwrites the same memory as x.
y === x           # true. op * v is lost
x = op * v        # Uses internal storage and passes pointer to x
y = op * x        # Silently overwrite x to zeros! Equivalent to mul!(x, A, x).
y == zeros(5)     # true. op * v and op * x are lost
```
"""
mutable struct PreallocatedLinearOperator{T, S, F, Ft, Fct} <: AbstractPreallocatedLinearOperator{T}
  nrow::Int
  ncol::Int
  symmetric::Bool
  hermitian::Bool
  prod!::F    # apply the operator to a vector
  tprod!::Ft   # apply the transpose operator to a vector
  ctprod!::Fct  # apply the transpose conjugate operator to a vector
  Mv::S # storage vector for prod!
  Mtu::S # storage vector for tprod!
  Maw::S # storage vector for ctprod!
  nprod::Int
  ntprod::Int
  nctprod::Int
end

PreallocatedLinearOperator{T}(nrow::Int, ncol::Int, symmetric::Bool, hermitian::Bool, 
                              prod!::F, tprod!::Ft, ctprod!::Fct, 
                              Mv::S, Mtu::S, Maw::S,
                              nprod::Int, ntprod::Int, nctprod::Int
                              ) where {T,S,F,Ft,Fct} = PreallocatedLinearOperator{T,S,F,Ft,Fct}(nrow, ncol, symmetric, hermitian, 
                                                                                                prod!, tprod!, ctprod!, 
                                                                                                Mv, Mtu, Maw,
                                                                                                nprod, ntprod, nctprod)

PreallocatedLinearOperator{T}(
  nrow::Int,
  ncol::Int,
  symmetric::Bool,
  hermitian::Bool,
  prod,
  tprod,
  ctprod,
  Mv::AbstractVector{T},
  Mtu::AbstractVector{T},
  Maw::AbstractVector{T}
) where {T} =
  PreallocatedLinearOperator{T}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod, Mv, Mtu, Maw, 0, 0, 0)

"""
    show(io, op)
Display basic information about a linear operator.
"""
function show(io::IO, op::AbstractPreallocatedLinearOperator)
  s = "Preallocated linear operator\n"
  s *= @sprintf("  nrow: %s\n", op.nrow)
  s *= @sprintf("  ncol: %d\n", op.ncol)
  s *= @sprintf("  eltype: %s\n", eltype(op))
  s *= @sprintf("  symmetric: %s\n", op.symmetric)
  s *= @sprintf("  hermitian: %s\n", op.hermitian)
  s *= @sprintf("  nprod:   %d\n", nprod(op))
  s *= @sprintf("  ntprod:  %d\n", ntprod(op))
  s *= @sprintf("  nctprod: %d\n", nctprod(op))
  s *= "\n"
  print(io, s)
end

# Constructors.
"""
    PreallocatedLinearOperator(Mv, Mtu, Maw, M; symmetric=false, hermitian=false)
Construct a linear operator from a dense or sparse matrix, using `Mv` as storage space
for `M * v` and `Mtu` as storage space for `transpose(M) * u` and `Maw` to store
`adjoint(M) * w`. Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian.
"""
function PreallocatedLinearOperator(
  Mv::AbstractVector{T},
  Mtu::AbstractVector{T},
  Maw::AbstractVector{T},
  M::AbstractMatrix{T};
  symmetric = false,
  hermitian = false,
) where {T}
  nrow, ncol = size(M)
  @assert length(Mv) == nrow
  @assert length(Mtu) == ncol
  @assert length(Maw) == ncol
  prod! = @closure (Mv, v, α, β) -> mul!(Mv, M, v, α, β)
  tprod! = @closure (Mtu, u, α, β) -> mul!(Mtu, transpose(M), u, α, β)
  ctprod! = @closure (Maw, w, α, β) -> mul!(Maw, adjoint(M), w, α, β)
  PreallocatedLinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, Mv, Mtu, Maw)
end

"""
    PreallocatedLinearOperator(Mv, M :: Symmetric{<:Real})
Construct a linear operator from a symmetric real square matrix `M` with preallocation
using `Mv` as storage space.
"""
PreallocatedLinearOperator(
  Mv::AbstractVector{T},
  M::Union{SymTridiagonal{T}, Symmetric{T}},
) where {T <: Real} = PreallocatedLinearOperator(Mv, Mv, Mv, M, symmetric = true, hermitian = true)

function PreallocatedLinearOperator(
  M::AbstractMatrix{T};
  storagetype = Vector{T},
  symmetric = false,
  hermitian = false,
) where {T}
  nrow, ncol = size(M)
  local Mv, Mtu, Maw
  if T <: Real
    if symmetric
      Maw = Mtu = Mv = storagetype(undef, nrow)
    else
      Mv = storagetype(undef, nrow)
      Maw = Mtu = storagetype(undef, ncol)
    end
  else
    if symmetric && hermitian # Actually real, but T is not
      Maw = Mtu = Mv = storagetype(undef, nrow)
    elseif symmetric
      Mtu = Mv = storagetype(undef, nrow)
      Maw = storagetype(undef, ncol)
    elseif hermitian
      Mv = storagetype(undef, nrow)
      Maw = Mtu = storagetype(undef, ncol)
    else
      Mv = storagetype(undef, nrow)
      Mtu = storagetype(undef, ncol)
      Maw = storagetype(undef, ncol)
    end
  end
  PreallocatedLinearOperator(Mv, Mtu, Maw, M, symmetric = symmetric, hermitian = hermitian)
end

"""
    PreallocatedLinearOperator(M)
Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function PreallocatedLinearOperator(
  M::SymTridiagonal{T};
  storagetype = Vector{T},
  kwargs...,
) where {T}
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  hermitian = eltype(M) <: Real
  Maw = hermitian ? Mv : storagetype(undef, ncol)
  PreallocatedLinearOperator(Mv, Mv, Maw, M, symmetric = true, hermitian = hermitian)
end

"""
    PreallocatedLinearOperator(M)
Constructs a linear operator from a symmetric matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function PreallocatedLinearOperator(M::Symmetric{T}; storagetype = Vector{T}, kwargs...) where {T}
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  hermitian = eltype(M) <: Real
  Maw = hermitian ? Mv : storagetype(undef, ncol)
  PreallocatedLinearOperator(Mv, Mv, Maw, M, symmetric = true, hermitian = hermitian)
end

"""
    PreallocatedLinearOperator(M)
Constructs a linear operator from a Hermitian matrix. If
its elements are real, it is also symmetric.
"""
function PreallocatedLinearOperator(M::Hermitian{T}; storagetype = Vector{T}, kwargs...) where {T}
  nrow, ncol = size(M)
  Mv = storagetype(undef, nrow)
  symmetric = eltype(M) <: Real
  Mtu = symmetric ? Mv : storagetype(undef, ncol)
  PreallocatedLinearOperator(Mv, Mtu, Mv, M, symmetric = symmetric, hermitian = true)
end