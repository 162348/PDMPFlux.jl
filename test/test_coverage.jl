# テストカバレッジの向上のための追加テスト

using Test
using PDMPFlux
using Random
using Statistics
using LinearAlgebra
using Distributions

@testset "Enhanced Coverage Tests" begin
    
    # @testset "Edge Cases and Boundary Conditions" begin
    #     # 非常に大きな値のテスト
    #     function U_large(x::AbstractVector)
    #         return 100 * x[1]^2
    #     end
        
    #     @testset "Large Potential" begin
    #         sampler = ZigZagAD(1, U_large, grid_size=0)
    #         output = sample_skeleton(sampler, 50, [0.0], [1.0], seed=42)
    #         @test length(output.t) > 0
    #         @test all(isfinite.(output.t))
    #     end
        
    #     # 負の値のテスト
    #     function U_negative(x::AbstractVector)
    #         return -x[1]^2 / 2
    #     end
        
    #     @testset "Negative Potential" begin
    #         sampler = ZigZagAD(1, U_negative, grid_size=0)
    #         output = sample_skeleton(sampler, 50, 0.0, 1.0, seed=42)
    #         @test length(output.t) > 0
    #         @test all(isfinite.(output.t))
    #     end
    # end
    
    @testset "High Dimensional Tests" begin
        # 高次元でのテスト
        for dim in [5, 10, 20]
            @testset "Dimension $dim" begin
                function U_high_dim(x::AbstractVector)
                    return sum(x.^2) / 2
                end
                
                sampler = ZigZagAD(dim, U_high_dim, grid_size=0)
                xinit = zeros(dim)
                vinit = ones(dim)
                
                output = sample_skeleton(sampler, 100, xinit, vinit, seed=42)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
                @test size(hcat(output.x...)) == (dim, length(output.t))
                @test size(hcat(output.v...)) == (dim, length(output.t))
            end
        end
    end
    
    @testset "Complex Potentials" begin
        # 複雑なポテンシャル関数のテスト
        function U_banana(x::AbstractVector)
            mean_x2 = (x[1]^2 - 1)
            return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
        end
        
        function U_funnel(x::AbstractVector)
            v = x[1]
            log_density_v = logpdf(Normal(0.0, 3.0), v)
            variance_other = exp(v)
            other_dim = length(x) - 1
            cov_other = I * variance_other
            mean_other = zeros(other_dim)
            log_density_other = logpdf(MvNormal(mean_other, cov_other), x[2:end])
            return - log_density_v - log_density_other
        end
        
        function U_ridged_gauss(x::AbstractVector)
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
        function U_test(x::AbstractVector)
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
                # `mean(..., dims=2)` returns a (dim, 1) matrix; compare as vectors.
                @test all(isapprox.(vec(mean(samples, dims=2)), zeros(dim); atol=0.5))
                @test all(isapprox.(vec(std(samples, dims=2)), ones(dim); atol=0.5))
            end
        end
    end
    
    @testset "Grid Size Variations" begin
        function U_test(x::AbstractVector)
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
        function U_test(x::AbstractVector)
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
        end
        
        # 長時間実行テスト
        @testset "Long Duration" begin
            sampler = ZigZagAD(dim, U_test, grid_size=0)
            output = sample_skeleton(sampler, 1000, xinit, vinit, seed=seed)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
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

    @testset "Uncovered samplers: SpeedUpZigZag / StickyZigZag" begin
        function U_quad(x::AbstractVector)
            return sum(abs2, x) / 2
        end

        @testset "SpeedUpZigZagAD basic run" begin
            dim = 2
            xinit = [0.0, 0.0]
            vinit = [1.0, 1.0]

            sampler = @test_logs (:warn,) SpeedUpZigZagAD(
                dim,
                U_quad;
                grid_size=5,
                tmax=0,                 # trigger adaptive tmax branch
                vectorized_bound=false, # trigger signed/vec compatibility branch
                signed_bound=true,
                AD_backend="Zygote",
            )
            @test sampler.adaptive == true
            @test sampler.signed_bound == false

            history = sample_skeleton(sampler, 80, xinit, vinit; seed=42, verbose=false)
            @test length(history.t) == 80
            @test all(isfinite.(history.t))
            samples = sample_from_skeleton(sampler, 50, history)
            @test size(samples) == (dim, 50)
            @test all(isfinite.(samples))
        end

        @testset "StickyZigZagAD stick/thaw loop run" begin
            dim = 2
            κ = fill(100.0, dim)  # large thawing rate to exercise thawing paths
            # make an axis crossing almost immediate for coord 1
            xinit = [1e-6, 0.2]
            vinit = [-1.0, -1.0]

            sampler = @test_logs (:warn,) StickyZigZagAD(
                dim,
                U_quad,
                κ;
                grid_size=5,
                tmax=0,                 # trigger adaptive tmax branch
                vectorized_bound=false, # trigger signed/vec compatibility branch
                signed_bound=true,
                AD_backend="Zygote",
            )
            @test sampler.adaptive == true
            @test sampler.signed_bound == false

            history = sample_skeleton(sampler, 120, xinit, vinit; seed=42, verbose=false)
            @test length(history.t) == 120
            @test all(isfinite.(history.t))

            # Sticky samplers should produce `is_active` information.
            @test size(history.is_active) == (dim, 120)
            @test any(.!history.is_active)  # at least one coordinate stuck at some point

            # Cover StickyPDMP overloads in `sample_from_skeleton`
            @test_throws ArgumentError sample_from_skeleton(sampler, 0, history)
            samplesN = sample_from_skeleton(sampler, 40, history)
            @test size(samplesN) == (dim, 40)
            @test all(isfinite.(samplesN))

            # dt-based reconstruction dispatch (also has StickyPDMP overload)
            samplesdt = sample_from_skeleton(sampler, 0.01, history)
            @test size(samplesdt, 1) == dim
            @test all(isfinite.(samplesdt))
        end

        @testset "Public API wrappers coverage (SamplingLoop.jl / sample.jl)" begin
            # 1D scalar dispatch paths
            function U_1d(x::Float64)
                return x^2 / 2
            end
            sampler1d = ZigZagAD(1, U_1d; grid_size=0)

            h1 = PDMPFlux.sample_skeleton(sampler1d, 30, 0.1, 1.0; seed=123, verbose=false)
            @test length(h1.t) == 30

            @test_throws ArgumentError PDMPFlux.sample_from_skeleton(sampler1d, 0, h1)
            s1 = PDMPFlux.sample(sampler1d, 30, 20, 0.1, 1.0; seed=123, verbose=false)
            @test size(s1) == (1, 20)
        end
    end
end
