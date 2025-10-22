# サンプラーのテスト
using Test
using Statistics
using LinearAlgebra

@testset "Samplers" begin
    
    @testset "ZigZag Sampler" begin
        # 1次元ガウシアンでのテスト
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        function ∇U_Gauss_1D(x::Float64)
            return x
        end
        
        dim = 1
        grid_size = 0  # constant bounds
        sampler = ZigZag(dim, ∇U_Gauss_1D, grid_size=grid_size)
        
        @test sampler.dim == dim
        @test sampler.grid_size == grid_size
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = 0.0
        vinit = 1.0
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test length(output.x) > 0
        @test length(output.v) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
        
        # サンプル生成のテスト
        N = 1000
        samples = sample_from_skeleton(sampler, N, output)
        
        @test size(samples, 1) == dim
        @test size(samples, 2) == N
        @test all(isfinite.(samples))
        
        # 統計的妥当性のテスト（ガウシアン分布の場合）
        sample_mean = mean(samples)
        sample_var = var(samples)
        @test abs(sample_mean) < 0.5  # 平均は0に近い
        @test 0.5 < sample_var < 2.0  # 分散は1に近い
    end
    
    @testset "ZigZagAD Sampler" begin
        # 1次元ガウシアンでのテスト
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        dim = 1
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_1D, grid_size=grid_size)
        
        @test sampler.dim == dim
        @test sampler.grid_size == grid_size
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = 0.0
        vinit = 1.0
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test length(output.x) > 0
        @test length(output.v) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
    end
    
    @testset "ForwardECMC Sampler" begin
        # 1次元ガウシアンでのテスト
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        dim = 1
        grid_size = 10
        sampler = ForwardECMCAD(dim, U_Gauss_1D, grid_size=grid_size)
        
        @test sampler.dim == dim
        @test sampler.grid_size == grid_size
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = 0.0
        vinit = 1.0
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test length(output.x) > 0
        @test length(output.v) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
    end
    
    @testset "BPS Sampler" begin
        # 2次元ガウシアンでのテスト
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = BPSAD(dim, U_Gauss_2D, grid_size=grid_size)
        
        @test sampler.dim == dim
        @test sampler.grid_size == grid_size
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test length(output.x) > 0
        @test length(output.v) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
    end
end
