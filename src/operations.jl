import Base.+, Base.-, Base.*, LinearAlgebra.mul!

function mul!(res::AbstractVector{T}, op::AbstractLinearOperator{T}, v::AbstractVector{T}, α::T, β::T) where T 
  (size(v, 1) == size(op, 2) && size(res, 1) == size(op, 1)) || throw(LinearOperatorException("shape mismatch"))
  increase_nprod(op)
  op.prod!(res, v, α, β)
end

function mul!(res::AbstractVector{T}, op::AbstractLinearOperator{T}, v::AbstractVector{T}) where T
  (size(v, 1) == size(op, 2) && size(res, 1) == size(op, 1)) || throw(LinearOperatorException("shape mismatch"))
  increase_nprod(op)
  op.prod!(res, v, one(T), zero(T))
end

# Apply an operator to a vector.
function *(op::AbstractLinearOperator{T}, v::AbstractVector{S}) where {T, S}
  nrow, ncol = size(op)
  res = similar(v, nrow)
  mul!(res, op, v)
  return res
end

# Unary operations.
+(op::AbstractLinearOperator) = op