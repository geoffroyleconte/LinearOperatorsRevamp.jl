export check_ctranspose, check_hermitian, check_positive_definite

"""
    check_ctranspose(op)
Cheap check that the operator and its conjugate transposed are related.
"""
function check_ctranspose(op::AbstractLinearOperator{T}) where {T <: Union{AbstractFloat, Complex}}
  (m, n) = size(op)
  x = rand(n)
  y = rand(m)
  yAx = dot(y, op * x)
  xAty = dot(x, op' * y)
  ε = eps(real(eltype(op)))
  return abs(yAx - conj(xAty)) < (abs(yAx) + ε) * ε^(1 / 3)
end

function check_ctranspose(op::AbstractLinearOperator{T}) where {T <: Integer}
  (m, n) = size(op)
  x = convert(Vector{T}, (floor.(10 * rand(n)))) .- 5
  y = convert(Vector{T}, (floor.(10 * rand(m)))) .- 5
  yAx = dot(y, op * x)
  xAty = dot(x, op' * y)
  return yAx == xAty
end

check_ctranspose(M::AbstractMatrix) = check_ctranspose(LinearOperator(M))

"""
    check_hermitian(op)
Cheap check that the operator is Hermitian.
"""
function check_hermitian(op::AbstractLinearOperator{T}) where {T <: Union{AbstractFloat, Complex}}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = rand(n)
  w = copy(op * v)  # copy necessary to guard against in-place operators
  s = dot(w, w)  # = (Av)'(Av) = v' A' A v.
  y = op * w
  t = dot(v, y)  # = v' A A v.
  ε = eps(real(eltype(op)))
  return abs(s - t) < (abs(s) + ε) * ε^(1 / 3)
end

function check_hermitian(op::AbstractLinearOperator{T}) where {T <: Integer}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = convert(Vector{T}, (floor.(10 * rand(n)))) .- 5
  w = copy(op * v)
  s = dot(w, w)  # = (Av)'(Av) = v' A' A v.
  y = op * w
  t = dot(v, y)  # = v' A A v.
  return s == t
end

check_hermitian(M::AbstractMatrix) = check_hermitian(LinearOperator(M))

"""
    check_positive_definite(op; semi=false)
Cheap check that the operator is positive (semi-)definite.
"""
function check_positive_definite(
  op::AbstractLinearOperator{T};
  semi = false,
) where {T <: Union{AbstractFloat, Complex}}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = rand(n)
  w = op * v
  vw = dot(v, w)
  ε = eps(real(eltype(op)))
  if imag(vw) > sqrt(ε) * abs(vw)
    return false
  end
  vw = real(vw)
  return semi ? (vw ≥ 0) : (vw > 0)
end

function check_positive_definite(op::AbstractLinearOperator{T}; semi = false) where {T <: Integer}
  m, n = size(op)
  m == n || throw(LinearOperatorException("shape mismatch"))
  v = convert(Vector{T}, (floor.(10 * rand(n)))) .- 5
  w = op * v
  vw = dot(v, w)
  return semi ? (vw ≥ 0) : (vw > 0)
end

check_positive_definite(M::AbstractMatrix; kwargs...) =
  check_positive_definite(LinearOperator(M); kwargs...)