function test_linop()
  (nrow, ncol) = (10, 6)
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)
  A1 = rand(nrow, ncol)
  opA1 = LinearOperator(A1)
  v = rand(ncol)
  res = zeros(nrow)

  # test mul3
  mul!(res, opA1, v) # build 
  opA1 = LinearOperator(A1)
  allocs = @allocated mul!(res, opA1, v)
  @test allocs == 0 
  res_true = zeros(nrow)
  mul!(res_true, A1, v)
  @test norm(res-res_true) ≤ sqrt(ϵ)

  # test mul5 
  α, β = 2.0, -3.0
  res = rand(nrow)
  res_true .= res
  opA1 = LinearOperator(A1)
  mul!(res, opA1, v, α, β)
  mul!(res, A1, v, α, β)
  res .= res_true
  allocs2 = @allocated mul!(res, opA1, v, α, β)
  println(allocs2)
  allocs = @allocated mul!(res_true, A1, v, α, β)
  println(allocs)
  @test allocs2 == 32 
  @test allocs == 0
  @test norm(res-res_true) ≤ sqrt(ϵ)

  # test prod 
  opA1 = LinearOperator(A1)
  res = opA1 * v 
  res_true = A1 * v 
  @test norm(res-res_true) ≤ sqrt(ϵ)
  
end

test_linop()