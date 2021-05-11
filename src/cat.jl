import Base.hcat, Base.vcat, Base.hvcat

hcat(A::AbstractLinearOperator, B::AbstractMatrix) = hcat(A, LinearOperator(B))

hcat(A::AbstractMatrix, B::AbstractLinearOperator) = hcat(LinearOperator(A), B)

function hcat_prod!(res::AbstractVector{T}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}, 
                    Ancol::Int, nV::Int,
                    v::AbstractVector{T}, α::T, β::T) where T
  A.prod!(res, view(v, 1:Ancol), α, β)
  B.prod!(res, view(v, (Ancol+1): nV), α, one(T))
end

function hcat_tprod!(res::AbstractVector{T}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T},
                     Ancol::Int, nV::Int,     
                     u::AbstractVector{T}, α::T, β::T) where T
  A.tprod!(view(res, 1:Ancol), u, α, β)
  B.tprod!(view(res, (Ancol+1): nV), u, α, β)
end

function hcat_ctprod!(res::AbstractVector{T}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T},
                      Ancol::Int, nV::Int,     
                      w::AbstractVector{T}, α::T, β::T) where T
  A.ctprod!(view(res, 1:Ancol), w, α, β)
  B.ctprod!(view(res, (Ancol+1): nV), w, α, β)
end
                     
function hcat(A::AbstractLinearOperator, B::AbstractLinearOperator)
  size(A, 1) == size(B, 1) || throw(LinearOperatorException("hcat: inconsistent row sizes"))

  nrow = size(A, 1)
  Ancol, Bncol = size(A, 2), size(B, 2)
  ncol = Ancol + Bncol
  S = promote_type(eltype(A), eltype(B))

  MvAB = similar(A.Mv)
  MtuAB = vcat(A.Mtu, B.Mtu)
  MawAB = vcat(A.Maw, B.Maw) 

  prod = @closure (res, v, α, β) -> hcat_prod!(res, A, B, Ancol, Ancol+Bncol, v, α, β)
  tprod = @closure (res, u, α, β) -> hcat_tprod!(res, A, B, Ancol, Ancol+Bncol, u, α, β)
  ctprod = @closure (res, w, α, β) -> hcat_tprod!(res, A, B, Ancol, Ancol+Bncol, w, α, β)
  LinearOperator{S}(nrow, ncol, false, false, prod, tprod, ctprod, MvAB, MtuAB, MawAB)
end

function hcat(ops::AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op ops[i]]
  end
  return op
end

vcat(A::AbstractLinearOperator, B::AbstractMatrix) = vcat(A, LinearOperator(B))

vcat(A::AbstractMatrix, B::AbstractLinearOperator) = vcat(LinearOperator(A), B)

function vcat_prod!(res::AbstractVector{T}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T},
                    Anrow::Int, nV::Int,     
                    u::AbstractVector{T}, α::T, β::T) where T
  A.prod!(view(res, 1:Anrow), u, α, β)
  B.prod!(view(res, (Anrow+1): nV), u, α, β)
end

function vcat_tprod!(res::AbstractVector{T}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}, 
                     Anrow::Int, nV::Int,
                     v::AbstractVector{T}, α::T, β::T) where T
  A.tprod!(res, view(v, 1:Anrow), α, β)
  B.tprod!(res, view(v, (Anrow+1): nV), α, one(T))
end

function vcat_ctprod!(res::AbstractVector{T}, A::AbstractLinearOperator{T}, B::AbstractLinearOperator{T}, 
                     Anrow::Int, nV::Int,
                     v::AbstractVector{T}, α::T, β::T) where T
  A.ctprod!(res, view(v, 1:Anrow), α, β)
  B.ctprod!(res, view(v, (Anrow+1): nV), α, one(T))
end

function vcat(A::AbstractLinearOperator, B::AbstractLinearOperator)
  size(A, 2) == size(B, 2) || throw(LinearOperatorException("vcat: inconsistent column sizes"))

  Anrow, Bnrow = size(A, 1), size(B, 1)
  nrow = Anrow + Bnrow
  ncol = size(A, 2)
  S = promote_type(eltype(A), eltype(B))

  MvAB = vcat(A.Mv, B.Mv)
  MtuAB = similar(A.Mtu)
  MawAB = similar(A.Maw)

  prod! = @closure (res, v, α, β) -> vcat_prod!(res, A, B, Anrow, Anrow+Bnrow, v, α, β)
  tprod! = @closure (res, u, α, β) -> vcat_tprod!(res, A, B, Anrow, Anrow+Bnrow, u, α, β)
  ctprod! = @closure (res, w, α, β) -> vcat_ctprod!(res, A, B, Anrow, Anrow+Bnrow, w, α, β)
  return LinearOperator{S}(nrow, ncol, false, false, prod!, tprod!, ctprod!, MvAB, MtuAB, MawAB)
end

function vcat(ops::AbstractLinearOperator...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]]
  end
  return op
end

# Removed by https://github.com/JuliaLang/julia/pull/24017
function hvcat(rows::Tuple{Vararg{Int}}, ops::AbstractLinearOperator...)
  nbr = length(rows)
  rs = Array{AbstractLinearOperator, 1}(undef, nbr)
  a = 1
  for i = 1:nbr
    rs[i] = hcat(ops[a:(a - 1 + rows[i])]...)
    a += rows[i]
  end
  vcat(rs...)
end