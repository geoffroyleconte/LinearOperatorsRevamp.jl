import Base.+, Base.-, Base.*, LinearAlgebra.mul!

function mul!(res::AbstractVector, op::AbstractLinearOperator{T}, v::AbstractVector, α, β) where T 
  (size(v, 1) == size(op, 2) && size(res, 1) == size(op, 1)) || throw(LinearOperatorException("shape mismatch"))
  increase_nprod(op)
  op.prod!(res, v, α, β)
end

function mul!(res::AbstractVector, op::AbstractLinearOperator{T}, v::AbstractVector) where T
  mul!(res, op, v, one(T), zero(T))
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

function -(op::AbstractLinearOperator{T}) where T
  prod! = @closure (res, v, α, β) -> op.prod!(res, v, -α, β)
  tprod! = @closure (res, u, α, β) -> op.tprod!(res, u, -α, β)
  ctprod! = @closure (res, w, α, β) -> op.ctprod!(res, w, -α, β)
  LinearOperator{T}(op.nrow, op.ncol, op.symmetric, op.hermitian, prod!, tprod!, ctprod!, op.Mv, op.Mtu, op.Maw)
end

function prod_op!(res::AbstractVector, op1::LinearOperator{T}, op2::LinearOperator{T}, 
                  v::AbstractVector, α, β) where T
  op2.prod!(op2.Mv, v, one(T), zero(T))
  op1.prod!(res, op2.Mv, α, β)
end

function tprod_op!(res::AbstractVector, op1::AbstractLinearOperator{T}, op2::AbstractLinearOperator{T}, 
                   u::AbstractVector, α, β) where T
  op1.tprod!(op1.Mtu, u, one(T), zero(T))
  op2.tprod!(res, op1.Mtu, α, β)
end

function ctprod_op!(res::AbstractVector, op1::AbstractLinearOperator{T}, op2::AbstractLinearOperator{T}, 
                    u::AbstractVector, α, β) where T
  op1.ctprod!(op1.Maw, w, one(T), zero(T))
  op2.ctprod!(res, op1.Maw, α, β)
end

## Operator times operator.
function *(op1::AbstractLinearOperator{T}, op2::AbstractLinearOperator{T}) where T
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    throw(LinearOperatorException("shape mismatch"))
  end
  prod! = @closure (res, v, α, β) -> prod_op!(res, op1, op2, v, α, β)
  tprod! = @closure (res, u, α, β) -> tprod_op!(res, op1, op2, u, α, β)
  ctprod! = @closure (res, w, α, β) -> ctprod_op!(res, op1, op2, w, α, β)
  LinearOperator{T}(m1, n2, false, false, prod!, tprod!, ctprod!, op1.Mv, op2.Mtu, op2.Maw)
end

## Matrix times operator.
*(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) * op
*(op::AbstractLinearOperator, M::AbstractMatrix) = op * LinearOperator(M)


## Scalar times operator. (# commutation α*v ???)
function *(op::AbstractLinearOperator, x::Number)
  S = promote_type(eltype(op), typeof(x))
  prod! = @closure (res, v, α, β) -> op.prod!(res, v, x * α, β)
  tprod! = @closure (res, u, α, β) -> op.tprod!(res, u, x * α, β)
  ctprod! = @closure (res, w, α, β) -> op.ctprod!(res, w, x' * α, β)
  LinearOperator{S}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x), prod!, tprod!, ctprod!, op.Mv, op.Mtu, op.Maw)
end

function *(x::Number, op::AbstractLinearOperator)
  return op * x
end

# Operator + operator.

function sum_prod!(res::AbstractVector, op1::AbstractLinearOperator{T}, op2::AbstractLinearOperator{T}, 
                   v::AbstractVector, α, β) where T
  op1.prod!(res, v, α, β)
  op2.prod!(res, v, α, one(T))
end

function sum_tprod!(res::AbstractVector, op1::AbstractLinearOperator{T}, op2::AbstractLinearOperator{T}, 
                    u::AbstractVector, α, β) where T
op1.tprod!(res, u, α, β)
op2.tprod!(res, u, α, one(T))
end

function sum_ctprod!(res::AbstractVector, op1::AbstractLinearOperator{T}, op2::AbstractLinearOperator{T}, 
                     w::AbstractVector, α, β) where T
  op1.ctprod!(res, w, α, β)
  op2.ctprod!(res, w, α, one(T))
end

function +(op1::AbstractLinearOperator, op2::AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  prod! = @closure (res, v, α, β) -> sum_prod!(res, op1, op2, v, α, β)
  tprod! = @closure (res, u, α, β) -> sum_tprod!(res, op1, op2, u, α, β)
  ctprod! = @closure (res, w, α, β) -> sum_ctprod!(res, op1, op2, w, α, β)
  return LinearOperator{S}(
    m1,
    n1,
    symmetric(op1) && symmetric(op2),
    hermitian(op1) && hermitian(op2),
    prod!,
    tprod!,
    ctprod!,
    op2.Mv,
    op2.Mtu,
    op2.Maw
  )
end

# Operator + matrix.
+(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) + op
+(op::AbstractLinearOperator, M::AbstractMatrix) = op + LinearOperator(M)

# # Operator .+ scalar.
# +(op::AbstractLinearOperator, x::Number) = op + x * opOnes(op.nrow, op.ncol)
# +(x::Number, op::AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
-(op1::AbstractLinearOperator, op2::AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
-(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) - op
-(op::AbstractLinearOperator, M::AbstractMatrix) = op - LinearOperator(M)

# # Operator - scalar.
# -(op::AbstractLinearOperator, x::Number) = op + (-x)
# -(x::Number, op::AbstractLinearOperator) = x + (-op)