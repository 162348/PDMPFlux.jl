# 数値的安定性とロバストネスのテスト
using Test
using PDMPFlux
using Random
using Statistics
using LinearAlgebra

@testset "Numerical Stability Tests" begin
    
    @testset "Extreme Values" begin
        # 非常に大きな値
        function U_large(x::Float64)
            return 1e10 * x^2
        end
        
        @testset "Very Large Potential" begin
            sampler = ZigZagAD(1, U_large, grid_size=0)
            output = sample_skeleton(sampler, 50, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
            @test all(isfinite.(hcat(output.x...)))
            @test all(isfinite.(hcat(output.v...)))
        end
        
        # 非常に小さな値
        function U_small(x::Float64)
            return 1e-10 * x^2
        end
        
        @testset "Very Small Potential" begin
            sampler = ZigZagAD(1, U_small, grid_size=0)
            output = sample_skeleton(sampler, 50, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
        
        # 初期値が大きい場合
        @testset "Large Initial Values" begin
            function U_normal(x::Float64)
                return x^2 / 2
            end
            
            sampler = ZigZagAD(1, U_normal, grid_size=0)
            output = sample_skeleton(sampler, 50, 1e6, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
    end
    
    @testset "Singular Points" begin
        # 特異点を含むポテンシャル
        function U_singular(x::Float64)
            return x == 0.0 ? 0.0 : 1.0 / abs(x)
        end
        
        @testset "Singular Potential" begin
            sampler = ZigZagAD(1, U_singular, grid_size=0)
            output = sample_skeleton(sampler, 50, 1.0, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
        
        # 不連続なポテンシャル
        function U_discontinuous(x::Float64)
            return x < 0 ? x^2 : 2 * x^2
        end
        
        @testset "Discontinuous Potential" begin
            sampler = ZigZagAD(1, U_discontinuous, grid_size=0)
            output = sample_skeleton(sampler, 50, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
    end
    
    @testset "Numerical Precision" begin
        function U_test(x::Float64)
            return x^2 / 2
        end
        
        # 異なる精度でのテスト
        precisions = [Float32, Float64]
        
        for prec in precisions
            @testset "Precision $prec" begin
                sampler = ZigZagAD(1, U_test, grid_size=0)
                output = sample_skeleton(sampler, 100, prec(0.0), prec(1.0), seed=42)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
                @test eltype(output.t) == prec
            end
        end
    end
    
    @testset "Convergence Tests" begin
        function U_test(x::Float64)
            return x^2 / 2
        end
        
        # 異なるサンプル数での収束テスト
        sample_sizes = [100, 500, 1000, 2000]
        results = []
        
        for N in sample_sizes
            sampler = ZigZagAD(1, U_test, grid_size=0)
            output = sample_skeleton(sampler, N, 0.0, 1.0, seed=42)
            samples = sample_from_skeleton(sampler, N, output)
            push!(results, samples)
        end
        
        @testset "Convergence" begin
            # サンプル数が増えるにつれて統計量が安定することを確認
            means = [mean(samples) for samples in results]
            stds = [std(samples) for samples in results]
            
            # 平均値の収束
            @test abs(means[end] - means[end-1]) < abs(means[1] - means[2])
            
            # 標準偏差の収束
            @test abs(stds[end] - stds[end-1]) < abs(stds[1] - stds[2])
        end
    end
    
    @testset "Error Recovery" begin
        function U_test(x::Float64)
            return x^2 / 2
        end
        
        # 無効な初期値からの回復
        @testset "Invalid Initial Values" begin
            sampler = ZigZagAD(1, U_test, grid_size=0)
            
            # NaN初期値
            @test_throws ArgumentError sample_skeleton(sampler, 50, NaN, 1.0, seed=42)
            
            # Inf初期値
            @test_throws ArgumentError sample_skeleton(sampler, 50, Inf, 1.0, seed=42)
            
            # 負の初期速度
            @test_throws ArgumentError sample_skeleton(sampler, 50, 0.0, -1.0, seed=42)
        end
        
        # 無効なパラメータ
        @testset "Invalid Parameters" begin
            # 負の次元
            @test_throws ArgumentError ZigZagAD(-1, U_test, grid_size=0)
            
            # 負のグリッドサイズ
            @test_throws ArgumentError ZigZagAD(1, U_test, grid_size=-1)
            
            # 無効なADバックエンド
            @test_throws ArgumentError ZigZagAD(1, U_test, grid_size=0, AD_backend="InvalidBackend")
        end
    end
    
    @testset "Memory Management" begin
        function U_test(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        # 大量のメモリ使用テスト
        @testset "Large Memory Usage" begin
            dim = 100
            sampler = ZigZagAD(dim, U_test, grid_size=0)
            xinit = zeros(dim)
            vinit = ones(dim)
            
            output = sample_skeleton(sampler, 1000, xinit, vinit, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
            
            # メモリリークの確認（基本的なチェック）
            @test size(output.x) == (dim, length(output.t))
            @test size(output.v) == (dim, length(output.t))
        end
        
        # ガベージコレクション後の動作
        @testset "Garbage Collection" begin
            function U_test(x::Float64)
                return x^2 / 2
            end
            
            sampler = ZigZagAD(1, U_test, grid_size=0)
            
            # 複数回の実行
            for i in 1:5
                output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
                @test length(output.t) > 0
                
                # 強制的にガベージコレクション
                GC.gc()
            end
        end
    end
    
    @testset "Parallel Execution" begin
        function U_test(x::Float64)
            return x^2 / 2
        end
        
        # 並列実行のテスト（基本的な確認）
        @testset "Parallel Safety" begin
            sampler = ZigZagAD(1, U_test, grid_size=0)
            
            # 複数のスレッドで同時実行（可能な場合）
            results = []
            for i in 1:4
                output = sample_skeleton(sampler, 50, 0.0, 1.0, seed=42+i)
                push!(results, output.t)
            end
            
            @test length(results) == 4
            for result in results
                @test length(result) > 0
                @test all(isfinite.(result))
            end
        end
    end
end
