# 診断機能のテスト
using Test

@testset "Diagnostics" begin
    
    @testset "Basic Diagnostic Tests" begin
        # 1次元ガウシアンでのテスト
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        dim = 1
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_1D, grid_size=grid_size)
        
        N_sk = 1000
        xinit = 0.0
        vinit = 1.0
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        # 診断関数がエラーなく実行されることをテスト
        @test_nowarn diagnostic(output)
        
        # 診断結果の基本的な妥当性をテスト
        # 実際の診断結果を取得する方法は実装に依存します
        # ここでは関数が正常に実行されることのみをテスト
    end
    
    @testset "Sample Statistics" begin
        # 2次元ガウシアンでのテスト
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=grid_size)
        
        N_sk = 1000
        N = 1000
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        samples = sample_from_skeleton(sampler, N, output)
        
        # サンプルの基本的な統計量をテスト
        @test size(samples, 1) == dim
        @test size(samples, 2) == N
        @test all(isfinite.(samples))
        
        # 平均が0に近いことをテスト（ガウシアン分布の場合）
        sample_mean = mean(samples, dims=2)
        @test abs(sample_mean[1]) < 1.0  # 緩い条件
        @test abs(sample_mean[2]) < 1.0  # 緩い条件
        
        # 分散が有限であることをテスト
        sample_var = var(samples, dims=2)
        @test all(isfinite.(sample_var))
        @test all(sample_var .> 0)
    end
    
    @testset "Convergence Tests" begin
        # 異なるサンプル数での一貫性テスト
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        dim = 1
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_1D, grid_size=grid_size)
        
        xinit = 0.0
        vinit = 1.0
        seed = 42
        
        # 異なるサンプル数でテスト
        N_sk_values = [100, 500, 1000]
        outputs = []
        
        for N_sk in N_sk_values
            output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
            push!(outputs, output)
            
            # 基本的な妥当性をテスト
        @test length(output.t) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
        end
        
        # サンプル数が増えるとイベント数も増えることをテスト
        @test length(outputs[3].times) >= length(outputs[2].times)
        @test length(outputs[2].times) >= length(outputs[1].times)
    end
end
