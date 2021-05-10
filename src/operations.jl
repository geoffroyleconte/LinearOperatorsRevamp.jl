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

function -(op::AbstractLinearOperator{T}) where {T}
  prod! = @closure (res, v, α, β) -> op.prod!(res, v, -α, β)
  tprod! = @closure (res, u, α, β) -> op.tprod(res, u, -α, β)
  ctprod! = @closure (res, w, α, β) -> op.ctprod!(res, w, -α, β)
  LinearOperator{T}(op.nrow, op.ncol, op.symmetric, op.hermitian, prod!, tprod!, ctprod!)
end

function *(op1::AbstractLinearOperator, op2::AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  prod = @closure v -> op1 * (op2 * v)
  tprod = @closure u -> transpose(op2) * (transpose(op1) * u)
  ctprod = @closure w -> op2' * (op1' * w)
  LinearOperator{S}(m1, n2, false, false, prod, tprod, ctprod)
end