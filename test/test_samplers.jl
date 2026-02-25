# サンプラーのテスト
using Test
using Statistics
using Random
using LinearAlgebra
using PDMPFlux

@testset "Samplers" begin
    
    @testset "ZigZag Sampler" begin
        # 1次元ガウシアンでのテスト
        function U_Gauss_1D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        function ∇U_Gauss_1D(x::AbstractVector)
            return [x[1]]
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
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
        
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

        # 時刻Tまでのスケルトンサンプリング（t[end] == T を保証）
        T = 1.0
        outputT = sample_skeleton(sampler, T, xinit, vinit, seed=seed, verbose=false, init_capacity=8)
        @test isapprox(outputT.t[end], T; atol=0, rtol=0)
        @test all(diff(outputT.t) .>= 0)  # 単調増加（同時刻は起こり得るので >=）
        samplesT = sample_from_skeleton(sampler, N, outputT)
        @test all(isfinite.(samplesT))
    end
    
    @testset "ZigZagAD Sampler" begin
        # 1次元ガウシアンでのテスト
        function U_Gauss_1D(x::AbstractVector)
            return sum(x.^2) / 2
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
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
    end
    
    @testset "ForwardECMC Sampler" begin
        # 3次元ガウシアンでのテスト
        function U_Gauss_3D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 3
        grid_size = 10
        sampler = ForwardECMCAD(dim, U_Gauss_3D, grid_size=grid_size)
        
        @test sampler.dim == dim
        @test sampler.grid_size == grid_size
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = randn(dim)
        vinit = randn(dim)
        vinit = vinit ./ norm(vinit)
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test length(output.x) > 0
        @test length(output.v) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
    end

    @testset "ForwardECMC: ∇U(x)=x (alias) must not mutate x" begin
        # Regression test: previously `_velocity_jump_event_chain` normalized ∇U(x) in-place.
        # If a user provided `∇U(x)=x`, that would corrupt the state `x`.
        let dim = 10
            ∇U_alias(x::Vector{Float64}) = x
            sampler = ForwardECMC(dim, ∇U_alias; grid_size=10, tmax=1.0, mix_p=0.0, switch=true)
            rng = MersenneTwister(123)
            x = randn(dim)
            x0 = copy(x)
            v = randn(dim); v ./= norm(v)
            v_new = sampler.velocity_jump(x, v, rng)
            @test x == x0
            @test all(isfinite.(v_new))
        end
    end

    @testset "Manual gradient: no AD should leak Dual into ∇U" begin
        # ForwardECMC: user-defined ∇U only for Vector{Float64}
        let dim = 10
            ∇U_strict(x::Vector{Float64}) = @. tanh(x / 2)
            sampler = ForwardECMC(dim, ∇U_strict; grid_size=10, tmax=1.0, mix_p=0.0, switch=true)
            xinit = randn(dim)
            vinit = ones(dim) / sqrt(dim)
            hist = sample_skeleton(sampler, 50, xinit, vinit; seed=1, verbose=false)
            @test length(hist.t) == 50
        end

        # ZigZag: vectorized bound path previously triggered ForwardDiff.derivative(t -> rate_vect(...))
        let dim = 5
            ∇U_strict(x::Vector{Float64}) = x
            sampler = ZigZag(dim, ∇U_strict; grid_size=10, tmax=1.0, vectorized_bound=true, signed_bound=true)
            xinit = randn(dim)
            vinit = ones(dim)
            hist = sample_skeleton(sampler, 50, xinit, vinit; seed=2, verbose=false)
            @test length(hist.t) == 50
        end

        # BPS: non-vectorized bound path previously triggered ForwardDiff.derivative(t -> rate(...))
        let dim = 2
            ∇U_strict(x::Vector{Float64}) = x
            sampler = BPS(dim, ∇U_strict; grid_size=10, tmax=1.0, refresh_rate=0.1)
            xinit = randn(dim)
            vinit = randn(dim); vinit ./= norm(vinit)
            hist = sample_skeleton(sampler, 50, xinit, vinit; seed=3, verbose=false)
            @test length(hist.t) == 50
        end
    end
    
    @testset "BPS Sampler" begin
        # 2次元ガウシアンでのテスト
        function U_Gauss_2D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = BPSAD(dim, U_Gauss_2D, grid_size=grid_size)
        
        @test sampler.dim == dim
        @test sampler.grid_size == grid_size
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = randn(dim)
        vinit = randn(dim)
        vinit = vinit ./ norm(vinit)
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test length(output.x) > 0
        @test length(output.v) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
    end

    @testset "RHMC Sampler" begin
        # 1次元ガウシアンでのテスト（exact-flow なら厳密、ここでは数値積分近似なので緩く検証）
        function ∇U_Gauss_1D(x::AbstractVector)
            return [x[1]]
        end

        dim = 1
        sampler = RHMC(
            dim,
            ∇U_Gauss_1D;
            mean_duration=1.0,
            phi=pi/2,
            step_size=0.01,
            tmax=10.0,
            adaptive=false,
        )

        N_sk = 2000
        xinit = 0.0
        vinit = 0.0
        seed = 123

        hist = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose=false)
        @test length(hist.t) == N_sk
        @test all(isfinite.(hist.t))
        @test all(isfinite.(hcat(hist.x...)))
        @test all(isfinite.(hcat(hist.v...)))

        N = 2000
        samples = sample_from_skeleton(sampler, N, hist)
        @test size(samples, 1) == dim
        @test size(samples, 2) == N
        @test all(isfinite.(samples))

        sample_mean = mean(samples)
        sample_var = var(samples)
        @test abs(sample_mean) < 0.5
        @test 0.5 < sample_var < 2.0
    end

    @testset "RHMCAD Sampler" begin
        function U_Gauss_1D(x::AbstractVector)
            return sum(x.^2) / 2
        end

        dim = 1
        sampler = RHMCAD(
            dim,
            U_Gauss_1D;
            AD_backend="Zygote",
            mean_duration=1.0,
            phi=pi/2,
            step_size=0.01,
            tmax=10.0,
            adaptive=false,
        )

        hist = sample_skeleton(sampler, 200, 0.0, 0.0, seed=456, verbose=false)
        @test length(hist.t) == 200
        @test all(isfinite.(hist.t))
        @test all(isfinite.(hcat(hist.x...)))
        @test all(isfinite.(hcat(hist.v...)))
    end
end
