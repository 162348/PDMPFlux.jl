"""
テストカバレッジの向上のための追加テスト
"""

using Test
using PDMPFlux
using Random
using Statistics
using LinearAlgebra
using Distributions

@testset "Enhanced Coverage Tests" begin
    
    @testset "Edge Cases and Boundary Conditions" begin
        # ゼロ次元のテスト
        function U_zero(x::Float64)
            return 0.0
        end
        
        @testset "Zero Potential" begin
            sampler = ZigZagAD(1, U_zero, grid_size=0)
            output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
        
        # 非常に大きな値のテスト
        function U_large(x::Float64)
            return 1e6 * x^2
        end
        
        @testset "Large Potential" begin
            sampler = ZigZagAD(1, U_large, grid_size=0)
            output = sample_skeleton(sampler, 50, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
        
        # 負の値のテスト
        function U_negative(x::Float64)
            return -x^2 / 2
        end
        
        @testset "Negative Potential" begin
            sampler = ZigZagAD(1, U_negative, grid_size=0)
            output = sample_skeleton(sampler, 50, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
    end
    
    @testset "High Dimensional Tests" begin
        # 高次元でのテスト
        for dim in [5, 10, 20]
            @testset "Dimension $dim" begin
                function U_high_dim(x::Vector{Float64})
                    return sum(x.^2) / 2
                end
                
                sampler = ZigZagAD(dim, U_high_dim, grid_size=0)
                xinit = zeros(dim)
                vinit = ones(dim)
                
                output = sample_skeleton(sampler, 100, xinit, vinit, seed=42)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
                @test size(output.x) == (dim, length(output.t))
                @test size(output.v) == (dim, length(output.t))
            end
        end
    end
    
    @testset "Complex Potentials" begin
        # 複雑なポテンシャル関数のテスト
        function U_banana(x::Vector{Float64})
            mean_x2 = (x[1]^2 - 1)
            return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
        end
        
        function U_funnel(x::Vector{Float64})
            d = length(x)
            v = x[1]
            return v^2 / 2 + (d-1) * log(v) + sum(x[2:end].^2) / (2 * v^2)
        end
        
        function U_ridged_gauss(x::Vector{Float64})
            return sum(x.^2) / 2 + 0.1 * sum(sin.(10 * x))
        end
        
        potentials = [
            ("Banana", U_banana, 3),
            ("Funnel", U_funnel, 3),
            ("RidgedGauss", U_ridged_gauss, 2)
        ]
        
        for (name, U, dim) in potentials
            @testset "$name Potential" begin
                sampler = ZigZagAD(dim, U, grid_size=0)
                xinit = zeros(dim)
                vinit = ones(dim)
                
                output = sample_skeleton(sampler, 100, xinit, vinit, seed=42)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
            end
        end
    end
    
    @testset "Sampler Comparison" begin
        function U_test(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        samplers = [
            ("ZigZag", ZigZagAD(dim, U_test, grid_size=0)),
            ("BPS", BPSAD(dim, U_test, grid_size=0)),
            ("Boomerang", BoomerangAD(dim, U_test, grid_size=0)),
        ]
        
        results = Dict()
        
        for (name, sampler) in samplers
            @testset "$name" begin
                output = sample_skeleton(sampler, 200, xinit, vinit, seed=seed)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
                
                # サンプル生成
                samples = sample_from_skeleton(sampler, 100, output)
                @test size(samples) == (dim, 100)
                @test all(isfinite.(samples))
                
                results[name] = samples
            end
        end
        
        # サンプラー間の比較（基本的な統計量）
        @testset "Sampler Comparison" begin
            for (name, samples) in results
                @test mean(samples, dims=2) ≈ zeros(dim) atol=0.5
                @test std(samples, dims=2) ≈ ones(dim) atol=0.5
            end
        end
    end
    
    @testset "Grid Size Variations" begin
        function U_test(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        grid_sizes = [0, 5, 10, 20]
        
        for grid_size in grid_sizes
            @testset "Grid Size $grid_size" begin
                sampler = ZigZagAD(dim, U_test, grid_size=grid_size)
                output = sample_skeleton(sampler, 100, xinit, vinit, seed=seed)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
            end
        end
    end
    
    @testset "Memory and Performance" begin
        function U_test(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 5
        xinit = zeros(dim)
        vinit = ones(dim)
        seed = 42
        
        # 大量のサンプリングテスト
        @testset "Large Sample Size" begin
            sampler = ZigZagAD(dim, U_test, grid_size=0)
            output = sample_skeleton(sampler, 10000, xinit, vinit, seed=seed)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
            
            # メモリ使用量の確認
            @test length(output.t) <= 10000
        end
        
        # 長時間実行テスト
        @testset "Long Duration" begin
            sampler = ZigZagAD(dim, U_test, grid_size=0)
            output = sample_skeleton(sampler, 1000, xinit, vinit, seed=seed, T=10.0)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
            @test maximum(output.t) <= 10.0
        end
    end
    
    @testset "Random Seed Variations" begin
        function U_test(x::Float64)
            return x^2 / 2
        end
        
        seeds = [1, 42, 123, 999, 2024]
        results = []
        
        for seed in seeds
            sampler = ZigZagAD(1, U_test, grid_size=0)
            output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=seed)
            push!(results, output.t)
        end
        
        # 異なるシードで異なる結果が得られることを確認
        @testset "Seed Reproducibility" begin
            for i in 1:length(seeds)
                for j in (i+1):length(seeds)
                    @test results[i] != results[j] || length(results[i]) == 0
                end
            end
        end
        
        # 同じシードで同じ結果が得られることを確認
        @testset "Seed Consistency" begin
            sampler = ZigZagAD(1, U_test, grid_size=0)
            output1 = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
            output2 = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
            @test output1.t ≈ output2.t
        end
    end
end
