struct Flip end

function (::Flip)(res, x, α, β)
  res .= (-α) .* x .+ β .* res
end

function test_callable()
  @testset ExtendedTestSet "Test callable" begin
    Mv = ones(2)
    op = LinearOperator(2, 2, true, true, Flip())
    @test op * ones(2) == -ones(2)
    @test op' * ones(2) == -ones(2)
    @test transpose(op) * ones(2) == -ones(2)
  end
end

test_callable()