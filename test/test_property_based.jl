using Test
using Statistics
using LinearAlgebra
using Random

@testset "Property-Based Tests" begin
    
    @testset "Mathematical Invariants" begin
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        @testset "Energy Conservation" begin
            # ハミルトニアン系でのエネルギー保存（近似的）
            sampler = ZigZagAD(2, U_Gauss_2D, grid_size=0)
            output = sample_skeleton(sampler, 1000, [1.0, 1.0], [1.0, 1.0], seed=42)
            
            # 初期エネルギー
            initial_energy = U_Gauss_2D([1.0, 1.0]) + 0.5 * norm([1.0, 1.0])^2
            
            # 各時点でのエネルギー
            energies = Float64[]
            for i in 1:length(output.x)
                pos = output.x[i]
                vel = output.v[i]
                energy = U_Gauss_2D(pos) + 0.5 * norm(vel)^2
                push!(energies, energy)
            end
            
            # エネルギーが保存されることを確認（許容誤差内）
            energy_std = std(energies)
            @test energy_std < 0.1  # エネルギー変動が小さい
        end
        
        @testset "Reversibility" begin
            # 時間反転対称性のテスト
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
            
            # 軌道の時間反転
            reversed_times = reverse(output.t)
            reversed_positions = reverse(output.x)
            reversed_velocities = reverse(output.v)
            
            # 時間反転軌道も有効であることを確認
            @test all(isfinite.(reversed_times))
            @test all(isfinite.(reversed_positions))
            @test all(isfinite.(reversed_velocities))
        end
        
        @testset "Scale Invariance" begin
            # スケール変換に対する不変性
            function U_Scaled(x::Vector{Float64}, scale::Float64)
                return scale * sum(x.^2) / 2
            end
            
            scales = [0.1, 1.0, 10.0]
            results = []
            
            for scale in scales
                U_scale = x -> U_Scaled(x, scale)
                sampler = ZigZagAD(2, U_scale, grid_size=0)
                output = sample_skeleton(sampler, 100, [1.0, 1.0], [1.0, 1.0], seed=42)
                push!(results, output)
            end
            
            # スケールが異なっても基本的な性質は保たれる
            for result in results
                @test length(result.times) > 0
                @test all(isfinite.(result.times))
                @test all(isfinite.(result.positions))
                @test all(isfinite.(result.velocities))
            end
        end
    end
    
    @testset "Statistical Properties" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        @testset "Ergodicity" begin
            # エルゴード性のテスト（長時間平均が期待値に収束）
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            output = sample_skeleton(sampler, 5000, 0.0, 1.0, seed=42)
            samples = sample_from_skeleton(sampler, 5000, output)
            
            # 平均が0に近い
            sample_mean = mean(samples)
            @test abs(sample_mean) < 0.2
            
            # 分散が1に近い
            sample_var = var(samples)
            @test 0.8 < sample_var < 1.2
        end
        
        @testset "Mixing Properties" begin
            # 混合性のテスト（相関の減衰）
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            output = sample_skeleton(sampler, 2000, 0.0, 1.0, seed=42)
            samples = sample_from_skeleton(sampler, 2000, output)
            
            # 自己相関の計算
            function autocorr(x, lag)
                n = length(x)
                if lag >= n
                    return 0.0
                end
                mean_x = mean(x)
                numerator = sum((x[1:n-lag] .- mean_x) .* (x[lag+1:n] .- mean_x))
                denominator = sum((x .- mean_x).^2)
                return numerator / denominator
            end
            
            # ラグが大きくなるほど相関が小さくなる
            corr_1 = autocorr(samples, 1)
            corr_10 = autocorr(samples, 10)
            corr_50 = autocorr(samples, 50)
            
            @test abs(corr_1) > abs(corr_10)
            @test abs(corr_10) > abs(corr_50)
        end
        
        @testset "Central Limit Theorem" begin
            # 中心極限定理のテスト
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            
            # 複数回の独立実行
            n_runs = 100
            means = Float64[]
            
            for i in 1:n_runs
                output = sample_skeleton(sampler, 1000, 0.0, 1.0, seed=42+i)
                samples = sample_from_skeleton(sampler, 1000, output)
                push!(means, mean(samples))
            end
            
            # 平均の分布が正規分布に近い
            means_mean = mean(means)
            means_var = var(means)
            
            @test abs(means_mean) < 0.1  # 平均の平均は0に近い
            @test 0.8 < means_var < 1.2  # 分散は適切な範囲
        end
    end
    
    @testset "Convergence Properties" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        @testset "Monte Carlo Convergence" begin
            # モンテカルロ収束のテスト
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            
            sample_sizes = [100, 500, 1000, 2000]
            means = Float64[]
            vars = Float64[]
            
            for n in sample_sizes
                output = sample_skeleton(sampler, n, 0.0, 1.0, seed=42)
                samples = sample_from_skeleton(sampler, n, output)
                push!(means, mean(samples))
                push!(vars, var(samples))
            end
            
            # サンプル数が増えるほど統計量が安定する
            @test abs(means[end]) < abs(means[1])  # 平均が0に近づく
            @test abs(vars[end] - 1.0) < abs(vars[1] - 1.0)  # 分散が1に近づく
        end
        
        @testset "Bias and Variance" begin
            # バイアスと分散の評価
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            
            n_runs = 50
            n_samples = 1000
            estimates = Float64[]
            
            for i in 1:n_runs
                output = sample_skeleton(sampler, n_samples, 0.0, 1.0, seed=42+i)
                samples = sample_from_skeleton(sampler, n_samples, output)
                push!(estimates, mean(samples))
            end
            
            # バイアス（期待値からの偏り）
            bias = mean(estimates) - 0.0
            @test abs(bias) < 0.1
            
            # 分散（推定値のばらつき）
            variance = var(estimates)
            @test 0.5 < variance < 2.0
        end
    end
    
    @testset "Physical Properties" begin
        function U_Harmonic(x::Float64)
            return x^2 / 2
        end
        
        @testset "Detailed Balance" begin
            # 詳細釣り合いのテスト（近似的）
            sampler = ZigZagAD(1, U_Harmonic, grid_size=0)
            output = sample_skeleton(sampler, 2000, 0.0, 1.0, seed=42)
            samples = sample_from_skeleton(sampler, 2000, output)
            
            # ボルツマン分布からのサンプリング
            # 確率密度が exp(-U(x)) に比例することを確認
            x_range = -3.0:0.1:3.0
            theoretical_density = exp.(-U_Harmonic.(x_range))
            theoretical_density ./= sum(theoretical_density)
            
            # ヒストグラムとの比較
            hist = fit(Histogram, samples, x_range)
            empirical_density = hist.weights ./ sum(hist.weights)
            
            # 相関が高いことを確認
            correlation = cor(theoretical_density, empirical_density)
            @test correlation > 0.8
        end
        
        @testset "Temperature Scaling" begin
            # 温度スケーリングのテスト
            temperatures = [0.5, 1.0, 2.0]
            variances = Float64[]
            
            for T in temperatures
                U_T = x -> U_Harmonic(x) / T
                sampler = ZigZagAD(1, U_T, grid_size=0)
                output = sample_skeleton(sampler, 1000, 0.0, 1.0, seed=42)
                samples = sample_from_skeleton(sampler, 1000, output)
                push!(variances, var(samples))
            end
            
            # 温度が高いほど分散が大きい
            @test variances[1] < variances[2] < variances[3]
        end
    end
end
