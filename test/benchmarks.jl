# パフォーマンスベンチマークテスト
# 注意: これらのテストは時間がかかるため、通常のテストスイートには含まれません

using BenchmarkTools
using PDMPFlux
using Random
using LinearAlgebra

# ベンチマーク用の関数
function benchmark_zigzag_1d()
    function U_Gauss_1D(x::AbstractVector)
        return sum(x.^2) / 2
    end
    
    dim = 1
    grid_size = 0
    sampler = ZigZagAD(dim, U_Gauss_1D, grid_size=grid_size)
    
    N_sk = 100_000
    xinit = randn(dim)[1]
    vinit = 1.0
    seed = 42
    
    return @benchmark sample_skeleton($sampler, $N_sk, $xinit, $vinit, seed=$seed)
end

function benchmark_zigzag_2d()
    function U_Gauss_2D(x::AbstractVector)
        return sum(x.^2) / 2
    end
    
    dim = 2
    grid_size = 0
    sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=grid_size)
    
    N_sk = 100_000
    xinit = randn(dim)
    vinit = [1.0, 1.0]
    seed = 42
    
    return @benchmark sample_skeleton($sampler, $N_sk, $xinit, $vinit, seed=$seed)
end

function benchmark_forwardecmc_3d()
    function U_Gauss_3D(x::AbstractVector)
        return sum(x.^2) / 2
    end
    
    dim = 3
    grid_size = 10
    sampler = ForwardECMCAD(dim, U_Gauss_3D, grid_size=grid_size)
    
    N_sk = 100_000
    xinit = randn(dim)
    vinit = randn(dim)
    vinit = vinit ./ norm(vinit)
    seed = 42
    
    return @benchmark sample_skeleton($sampler, $N_sk, $xinit, $vinit, seed=$seed)
end

function benchmark_ad_backends()
    function U_banana(x::AbstractVector)
        mean_x2 = (x[1]^2 - 1)
        return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
    end
    
    dim = 10
    grid_size = 0
    N_sk = 10_000
    xinit = ones(dim)
    vinit = ones(dim)
    seed = 42
    
    backends = ["ForwardDiff", "Zygote", "Enzyme"]
    results = Dict()
    
    for backend in backends
        sampler = ZigZagAD(dim, U_banana, grid_size=grid_size, AD_backend=backend)
        results[backend] = @benchmark sample_skeleton($sampler, $N_sk, $xinit, $vinit, seed=$seed)
    end
    
    return results
end

# ベンチマーク実行関数
function run_benchmarks()
    println("Running PDMPFlux benchmarks...")
    
    println("\n1D ZigZag benchmark:")
    result_1d = benchmark_zigzag_1d()
    println(result_1d)
    
    println("\n2D ZigZag benchmark:")
    result_2d = benchmark_zigzag_2d()
    println(result_2d)
    
    println("\n3D ForwardECMC benchmark:")
    result_fecmc = benchmark_forwardecmc_3d()
    println(result_fecmc)
    
    println("\nAD Backend comparison:")
    results_ad = benchmark_ad_backends()
    for (backend, result) in results_ad
        println("\n$backend:")
        println(result)
    end
end

# ベンチマークを実行する場合は以下のコメントを外してください
# run_benchmarks()
