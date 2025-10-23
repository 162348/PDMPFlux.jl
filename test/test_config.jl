"""
テスト設定とユーティリティ関数
"""

# テスト設定
const TEST_CONFIG = Dict(
    :quick_samples => 50,
    :standard_samples => 500,
    :extended_samples => 1000,
    :quick_timeout => 60,
    :standard_timeout => 300,
    :extended_timeout => 600,
    :tolerance => 1e-6,
    :seed => 42
)

# テスト用のポテンシャル関数
module TestPotentials
    export U_Gauss_1D, U_Gauss_2D, U_Gauss_3D, U_banana, U_funnel, U_ridged_gauss
    
    function U_Gauss_1D(x::Float64)
        return x^2 / 2
    end
    
    function U_Gauss_2D(x::AbstractVector)
        return sum(x.^2) / 2
    end
    
    function U_Gauss_3D(x::AbstractVector)
        return sum(x.^2) / 2
    end
    
    function U_banana(x::AbstractVector)
        mean_x2 = (x[1]^2 - 1)
        return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
    end
    
    function U_funnel(x::AbstractVector)
        d = length(x)
        v = x[1]
        return v^2 / 2 + (d-1) * log(v) + sum(x[2:end].^2) / (2 * v^2)
    end
    
    function U_ridged_gauss(x::AbstractVector)
        return sum(x.^2) / 2 + 0.1 * sum(sin.(10 * x))
    end
end

# テスト用のユーティリティ関数
module TestUtils
    export setup_test_environment, cleanup_test_environment, run_with_timeout
    
    function setup_test_environment()
        Random.seed!(TEST_CONFIG[:seed])
        return nothing
    end
    
    function cleanup_test_environment()
        GC.gc()
        return nothing
    end
    
    function run_with_timeout(f, timeout_seconds)
        # 基本的なタイムアウト実装（実際の実装は環境に依存）
        try
            return f()
        catch e
            if isa(e, InterruptException)
                throw(ErrorException("Test timed out after $timeout_seconds seconds"))
            else
                rethrow(e)
            end
        end
    end
end

# テスト用のアサーション関数
module TestAssertions
    export assert_finite, assert_positive, assert_approximately_equal
    
    function assert_finite(x, name="value")
        @assert all(isfinite.(x)) "$name must be finite"
    end
    
    function assert_positive(x, name="value")
        @assert all(x .> 0) "$name must be positive"
    end
    
    function assert_approximately_equal(x, y, tol=TEST_CONFIG[:tolerance], name="values")
        @assert all(abs.(x .- y) .< tol) "$name must be approximately equal (tolerance: $tol)"
    end
end
