# 統合テスト - 実際の使用例とワークフローのテスト
using Test
using Statistics
using LinearAlgebra
using Random
using PDMPFlux

@testset "Integration Tests" begin
    
    @testset "Real-World Workflows" begin
        @testset "Bayesian Inference Workflow" begin
            # ベイズ推論の典型的なワークフロー
            function U_Posterior(x::AbstractVector)
                # 事前分布: N(0, 1)
                prior = -sum(x.^2) / 2
                
                # 尤度: 観測データ y = 1.0, ノイズ分散 = 0.5
                y_obs = 1.0
                noise_var = 0.5
                likelihood = -(x[1] - y_obs)^2 / (2 * noise_var)
                
                return -(prior + likelihood)
            end
            
            # サンプリング
            dim = 1
            sampler = ZigZagAD(dim, U_Posterior, grid_size=0)
            output = sample_skeleton(sampler, 2000, 0.0, 1.0, seed=42)
            samples = sample_from_skeleton(sampler, 2000, output)
            
            # 事後分布の統計量
            posterior_mean = mean(samples)
            posterior_var = var(samples)
            
            # 理論値との比較（共役事前分布の場合）
            # 事後平均 ≈ 0.67, 事後分散 ≈ 0.33
            @test 0.5 < posterior_mean < 0.8
            @test 0.2 < posterior_var < 0.5
        end
        
        @testset "High-Dimensional Sampling" begin
            # 高次元ガウシアン分布のサンプリング
            function U_HighDim(x::AbstractVector)
                return sum(x.^2) / 2
            end
            
            dim = 20
            sampler = ForwardECMCAD(dim, U_HighDim, grid_size=0)
            xinit = zeros(dim)
            vinit = ones(dim)
            
            output = sample_skeleton(sampler, 1000, xinit, vinit, seed=42)
            samples = sample_from_skeleton(sampler, 1000, output)
            
            # 高次元での統計的性質
            @test size(samples) == (dim, 1000)
            @test all(isfinite.(samples))
            
            # 各次元の平均と分散
            sample_means = mean(samples, dims=2)
            sample_vars = var(samples, dims=2)
            
            @test all(abs.(sample_means) .< 0.5)
            @test all(0.5 .< sample_vars .< 2.0)
        end
        
        @testset "Multi-Modal Distribution" begin
            # 多峰分布のサンプリング
            function U_Mixture(x::AbstractVector)
                # 2つのガウシアンの混合
                x1 = x[1]
                mode1 = -((x1 - 2.0)^2) / 2
                mode2 = -((x1 + 2.0)^2) / 2
                return -log(exp(mode1) + exp(mode2))
            end
            
            sampler = ZigZagAD(1, U_Mixture, grid_size=0)
            output = sample_skeleton(sampler, 3000, 0.0, 1.0, seed=42)
            samples = sample_from_skeleton(sampler, 3000, output)
            
            # 両方のモードがサンプリングされることを確認
            positive_samples = sum(samples .> 0)
            negative_samples = sum(samples .< 0)
            
            @test positive_samples > 100  # 正のモード
            @test negative_samples > 100  # 負のモード
        end
    end
    
    @testset "Cross-Sampler Comparisons" begin
        function U_Gauss_2D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 3
        N_sk = 1000
        N = 1000
        xinit = randn(dim)
        vinit = ones(dim)
        seed = 42
        
        # 異なるサンプラーでの結果比較
        samplers = [
            ("ZigZag", ZigZagAD(dim, U_Gauss_2D, grid_size=0)),
            ("BPS", BPSAD(dim, U_Gauss_2D, grid_size=0)),
            ("ForwardECMC", ForwardECMCAD(dim, U_Gauss_2D, grid_size=10)),
        ]
        
        results = Dict()
        
        for (name, sampler) in samplers
            if name in ["BPS", "ForwardECMC"]
                vinit = vinit ./ norm(vinit)
            end
            output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
            samples = sample_from_skeleton(sampler, N, output)
            results[name] = samples
        end
        
        # 各サンプラーの結果が統計的に類似していることを確認
        for (name, samples) in results
            sample_mean = mean(samples, dims=2)
            sample_var = var(samples, dims=2)
            
            @test all(abs.(sample_mean) .< 0.5)
            @test all(0.0 .< sample_var .< 2.0)
        end
    end
    
    @testset "Performance Integration" begin
        function U_Gauss_2D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        @testset "Memory Usage" begin
            # メモリ使用量のテスト
            dim = 2
            sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=0)
            
            # 大量のサンプル生成
            output = sample_skeleton(sampler, 5000, [0.0, 0.0], [1.0, 1.0], seed=42)
            samples = sample_from_skeleton(sampler, 5000, output)
            
            @test size(samples) == (2, 5000)
            @test all(isfinite.(samples))
            
            # メモリが適切に管理されていることを確認
            GC.gc()
            @test true  # エラーが発生しなければ成功
        end
        
        @testset "Computational Efficiency" begin
            # 計算効率のテスト
            dim = 2
            sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=0)
            
            # 実行時間の測定（相対的）
            start_time = time()
            output = sample_skeleton(sampler, 1000, [0.0, 0.0], [1.0, 1.0], seed=42)
            samples = sample_from_skeleton(sampler, 1000, output)
            end_time = time()
            
            execution_time = end_time - start_time
            
            # 合理的な実行時間内で完了することを確認
            @test execution_time < 10.0  # 10秒以内
            @test length(output.t) > 100  # 十分なイベントが生成される
        end
    end
    
    @testset "Diagnostic Integration" begin
        function U_Gauss_3D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        @testset "Diagnostic Workflow" begin
            # 診断機能との統合テスト
            sampler = ForwardECMCAD(3, U_Gauss_3D, grid_size=0)
            xinit = randn(3)
            vinit = randn(3)
            vinit = vinit ./ norm(vinit)
            output = sample_skeleton(sampler, 1000, xinit, vinit, seed=42)
            
            # 診断関数がエラーなく実行される
            @test_nowarn diagnostic(output)
            
            # プロット関数がエラーなく実行される
            @test_nowarn plot_traj(output, 100)
            
            # サンプル生成とプロット
            samples = sample_from_skeleton(sampler, 1000, output)
            @test_nowarn jointplot(samples)
        end
        
        @testset "Visualization Pipeline" begin
            # 可視化パイプラインのテスト
            sampler = ForwardECMCAD(3, U_Gauss_3D, grid_size=0)
            xinit = randn(3)
            vinit = randn(3)
            vinit = vinit ./ norm(vinit)
            output = sample_skeleton(sampler, 500, xinit, vinit, seed=42)
            samples = sample_from_skeleton(sampler, 500, output)
            
            # 各種プロットがエラーなく実行される
            @test_nowarn plot_traj(output, 100)
            @test_nowarn plot_traj(output, 100, plot_type="3D")
            @test_nowarn jointplot(samples)
            @test_nowarn marginalplot(samples, d=1)
            
            # アニメーション（短時間版）
            @static if SKIP_GIF_TEST
                @test true  # 空のテストセット回避用のダミー
            else
                @test begin
                    result = anim_traj(output, 10, filename="test_integration.gif")
                    return result !== nothing && isfile("test_integration.gif")
                end
            end
            
            # テスト用ファイルのクリーンアップ
            if isfile("test_integration.gif")
                rm("test_integration.gif")
            end
        end
    end
    
    @testset "Error Recovery" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        @testset "Graceful Degradation" begin
            # エラーからの回復テスト
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            
            # 通常の実行
            output1 = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
            @test length(output1.t) > 0
            
            # 異なるパラメータでの実行
            output2 = sample_skeleton(sampler, 100, 1.0, -1.0, seed=43)
            @test length(output2.t) > 0
            
            # 両方とも有効な結果を返す
            @test all(isfinite.(output1.t))
            @test all(isfinite.(output2.t))
        end
        
        @testset "Robustness to Parameters" begin
            # パラメータの変化に対する頑健性
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            
            # 様々な初期条件でのテスト
            initial_conditions = [
                (0.0, 1.0),
                (1.0, 1.0),
                (-1.0, 1.0),
                (0.0, -1.0),
                (2.0, 0.5),
            ]
            
            for (x0, v0) in initial_conditions
                output = sample_skeleton(sampler, 100, x0, v0, seed=42)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
                @test all(isfinite.(hcat(output.x...)))
                @test all(isfinite.(hcat(output.v...)))
            end
        end
    end
end
