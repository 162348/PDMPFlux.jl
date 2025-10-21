"""
テスト用のヘルパー関数とマクロ
"""

using Test
using PDMPFlux
using Random
using Statistics
using LinearAlgebra

# テスト用のマクロ
macro test_approximately_equal(x, y, tol=1e-6, name="values")
    quote
        @test all(abs.($x .- $y) .< $tol) "$name must be approximately equal (tolerance: $tol)"
    end
end

macro test_finite(x, name="value")
    quote
        @test all(isfinite.($x)) "$name must be finite"
    end
end

macro test_positive(x, name="value")
    quote
        @test all($x .> 0) "$name must be positive"
    end
end

# テスト用の関数
function create_test_sampler(dim::Int, potential_type::String="gaussian", grid_size::Int=0, AD_backend::String="ForwardDiff")
    if potential_type == "gaussian"
        if dim == 1
            U(x::Float64) = x^2 / 2
        else
            U(x::Vector{Float64}) = sum(x.^2) / 2
        end
    elseif potential_type == "banana"
        U(x::Vector{Float64}) = begin
            mean_x2 = (x[1]^2 - 1)
            -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
        end
    elseif potential_type == "funnel"
        U(x::Vector{Float64}) = begin
            d = length(x)
            v = x[1]
            v^2 / 2 + (d-1) * log(v) + sum(x[2:end].^2) / (2 * v^2)
        end
    else
        error("Unknown potential type: $potential_type")
    end
    
    return ZigZagAD(dim, U, grid_size=grid_size, AD_backend=AD_backend)
end

function create_test_initial_conditions(dim::Int, x_type::String="zero", v_type::String="ones")
    if x_type == "zero"
        xinit = zeros(dim)
    elseif x_type == "ones"
        xinit = ones(dim)
    elseif x_type == "random"
        xinit = randn(dim)
    else
        error("Unknown x_type: $x_type")
    end
    
    if v_type == "ones"
        vinit = ones(dim)
    elseif v_type == "random"
        vinit = randn(dim)
    elseif v_type == "alternating"
        vinit = [(-1)^i for i in 1:dim]
    else
        error("Unknown v_type: $v_type")
    end
    
    return xinit, vinit
end

function run_sampler_test(sampler, N::Int, xinit, vinit; seed::Int=42, T::Float64=Inf)
    output = sample_skeleton(sampler, N, xinit, vinit, seed=seed, T=T)
    
    # 基本的な検証
    @test length(output.t) > 0
    @test all(isfinite.(output.t))
    @test all(isfinite.(output.x))
    @test all(isfinite.(output.v))
    
    # 次元の確認
    dim = length(xinit)
    @test size(output.x) == (dim, length(output.t))
    @test size(output.v) == (dim, length(output.t))
    
    return output
end

function test_sampler_reproducibility(sampler, N::Int, xinit, vinit; seed::Int=42, num_runs::Int=3)
    results = []
    for i in 1:num_runs
        output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
        push!(results, output.t)
    end
    
    # 同じシードで同じ結果が得られることを確認
    for i in 2:num_runs
        @test results[1] ≈ results[i]
    end
    
    return results
end

function test_sampler_statistics(sampler, N::Int, xinit, vinit; seed::Int=42, expected_mean=nothing, expected_std=nothing, tol::Float64=0.5)
    output = run_sampler_test(sampler, N, xinit, vinit, seed=seed)
    samples = sample_from_skeleton(sampler, N, output)
    
    # 統計量の計算
    sample_mean = mean(samples, dims=2)[:]
    sample_std = std(samples, dims=2)[:]
    
    # 期待値との比較
    if expected_mean !== nothing
        @test_approximately_equal(sample_mean, expected_mean, tol, "sample mean")
    end
    
    if expected_std !== nothing
        @test_approximately_equal(sample_std, expected_std, tol, "sample std")
    end
    
    return samples, sample_mean, sample_std
end

function benchmark_sampler(sampler, N::Int, xinit, vinit; seed::Int=42, num_runs::Int=5)
    times = []
    for i in 1:num_runs
        start_time = time()
        output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
        end_time = time()
        push!(times, end_time - start_time)
    end
    
    avg_time = mean(times)
    std_time = std(times)
    
    println("Sampler benchmark: $(avg_time) ± $(std_time) seconds for $N samples")
    
    return avg_time, std_time
end

# テスト用のデータ生成
function generate_test_data(dim::Int, N::Int, distribution::String="gaussian"; seed::Int=42)
    Random.seed!(seed)
    
    if distribution == "gaussian"
        return randn(dim, N)
    elseif distribution == "uniform"
        return 2 * rand(dim, N) .- 1
    elseif distribution == "exponential"
        return randexp(dim, N)
    else
        error("Unknown distribution: $distribution")
    end
end

# テスト用の比較関数
function compare_samplers(samplers::Vector, N::Int, xinit, vinit; seed::Int=42)
    results = Dict()
    
    for (name, sampler) in samplers
        output = run_sampler_test(sampler, N, xinit, vinit, seed=seed)
        samples = sample_from_skeleton(sampler, N, output)
        results[name] = samples
    end
    
    return results
end

# テスト用の可視化関数（テスト用の簡易版）
function plot_test_results(results::Dict; title::String="Test Results")
    # 実際のプロットは行わず、データの確認のみ
    for (name, samples) in results
        println("$name: mean=$(mean(samples)), std=$(std(samples))")
    end
end

# テスト用のエラーハンドリング
function test_error_handling(f, expected_error_type=nothing)
    if expected_error_type === nothing
        @test_nowarn f()
    else
        @test_throws expected_error_type f()
    end
end

# テスト用のパフォーマンス測定
function measure_performance(f, num_runs::Int=3)
    times = []
    for i in 1:num_runs
        start_time = time()
        result = f()
        end_time = time()
        push!(times, end_time - start_time)
    end
    
    return mean(times), std(times), result
end

# テスト用のメモリ使用量測定
function measure_memory_usage(f)
    # 基本的なメモリ使用量の測定
    start_memory = Base.gc_bytes()
    result = f()
    end_memory = Base.gc_bytes()
    
    return end_memory - start_memory, result
end
