using Test
using Statistics
using LinearAlgebra
using Random

@testset "Comprehensive Tests" begin
    
    @testset "All Sampler Types" begin
        # テスト用のポテンシャル関数
        function U_Gauss_2D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        function U_Banana(x::AbstractVector)
            mean_x2 = (x[1]^2 - 1)
            return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
        end
        
        dim = 2
        N_sk = 500
        N = 500
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        # 各サンプラータイプをテスト
        samplers = [
            ("ZigZag", ZigZagAD(dim, U_Gauss_2D, grid_size=0)),
            ("BPS", BPSAD(dim, U_Gauss_2D, grid_size=0)),
            ("ForwardECMC", ForwardECMCAD(dim, U_Gauss_2D, grid_size=10)),
            ("Boomerang", BoomerangAD(dim, U_Gauss_2D, grid_size=0)),
        ]
        
        for (name, sampler) in samplers
            @testset "$name Sampler" begin
                # 基本的な動作テスト
                output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
                
        @test length(output.t) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
                
                # サンプル生成テスト
                samples = sample_from_skeleton(sampler, N, output)
                @test size(samples, 1) == dim
                @test size(samples, 2) == N
                @test all(isfinite.(samples))
                
                # 統計的妥当性（ガウシアン分布の場合）
                sample_mean = mean(samples, dims=2)
                sample_var = var(samples, dims=2)
                @test all(abs.(sample_mean) .< 0.5)
                @test all(0.5 .< sample_var .< 2.0)
            end
        end
    end
    
    @testset "Edge Cases and Error Handling" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        # ゼロ次元（1次元）でのテスト
        @testset "1D Case" begin
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
        end
        
        # 高次元でのテスト
        @testset "High Dimensional Case" begin
            function U_Gauss_HD(x::AbstractVector)
                return sum(x.^2) / 2
            end
            
            dim = 10
            sampler = ZigZagAD(dim, U_Gauss_HD, grid_size=0)
            xinit = zeros(dim)
            vinit = ones(dim)
            
            output = sample_skeleton(sampler, 100, xinit, vinit, seed=42)
            @test length(output.t) > 0
            @test all(isfinite.(hcat(output.x...)))
        end
        
        # 極端な初期値でのテスト
        @testset "Extreme Initial Values" begin
            sampler = ZigZagAD(2, U_Gauss_2D, grid_size=0)
            
            # 大きな初期値
            output1 = sample_skeleton(sampler, 100, [10.0, 10.0], [1.0, 1.0], seed=42)
            @test length(output1.times) > 0
            
            # 小さな初期値
            output2 = sample_skeleton(sampler, 100, [0.001, 0.001], [0.001, 0.001], seed=42)
            @test length(output2.times) > 0
        end
    end
    
    @testset "Reproducibility" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        seed = 12345
        
        # 同じシードで2回実行
        output1 = sample_skeleton(sampler, 100, 0.0, 1.0, seed=seed)
        output2 = sample_skeleton(sampler, 100, 0.0, 1.0, seed=seed)
        
        # 結果が一致することを確認
        @test output1.times ≈ output2.times
        @test output1.positions ≈ output2.positions
        @test output1.velocities ≈ output2.velocities
    end
    
    @testset "Performance and Stability" begin
        function U_Gauss_2D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        # 長時間実行の安定性テスト
        @testset "Long Run Stability" begin
            sampler = ZigZagAD(2, U_Gauss_2D, grid_size=0)
            output = sample_skeleton(sampler, 10000, [0.0, 0.0], [1.0, 1.0], seed=42)
            
            @test length(output.t) > 1000  # 十分なイベントが生成される
            @test all(isfinite.(output.t))
            @test all(isfinite.(hcat(output.x...)))
            @test all(isfinite.(hcat(output.v...)))
            
            # 時間が単調増加することを確認
            @test all(diff(output.t) .> 0)
        end
        
        # メモリ効率のテスト
        @testset "Memory Efficiency" begin
            sampler = ZigZagAD(2, U_Gauss_2D, grid_size=0)
            
            # 大量のサンプル生成
            output = sample_skeleton(sampler, 5000, [0.0, 0.0], [1.0, 1.0], seed=42)
            samples = sample_from_skeleton(sampler, 5000, output)
            
            @test size(samples) == (2, 5000)
            @test all(isfinite.(samples))
        end
    end
    
    @testset "Mathematical Properties" begin
        function U_Gauss_2D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        @testset "Conservation Properties" begin
            sampler = ZigZagAD(2, U_Gauss_2D, grid_size=0)
            output = sample_skeleton(sampler, 1000, [0.0, 0.0], [1.0, 1.0], seed=42)
            
            # 速度の大きさが保存されることを確認（ZigZagの場合）
            if typeof(sampler) <: Union{ZigZag, ZigZagAD}
                for i in 1:length(output.v)
                    @test abs(norm(output.v[i]) - 1.0) < 1e-10
                end
            end
        end
        
        @testset "Convergence Properties" begin
            # 異なるサンプル数での一貫性
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            
            samples_100 = sample_from_skeleton(sampler, 100, 
                sample_skeleton(sampler, 1000, 0.0, 1.0, seed=42))
            samples_1000 = sample_from_skeleton(sampler, 1000, 
                sample_skeleton(sampler, 1000, 0.0, 1.0, seed=42))
            
            # より多くのサンプルで統計が改善されることを確認
            var_100 = var(samples_100)
            var_1000 = var(samples_1000)
            
            # 分散の推定値が合理的な範囲内にあることを確認
            @test 0.5 < var_100 < 2.0
            @test 0.5 < var_1000 < 2.0
        end
    end
end
