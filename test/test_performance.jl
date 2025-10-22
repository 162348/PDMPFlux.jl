# パフォーマンステストとベンチマーク

using Test
using PDMPFlux
using Random
using Statistics
using LinearAlgebra
using BenchmarkTools

@testset "Performance Tests" begin
    
    @testset "Basic Performance" begin
        function U_test(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 2
        sampler = ZigZagAD(dim, U_test, grid_size=0)
        xinit = zeros(dim)
        vinit = ones(dim)
        seed = 42
        
        # 基本的なパフォーマンステスト
        @testset "Small Sample Size" begin
            N = 100
            start_time = time()
            output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
            end_time = time()
            
            elapsed_time = end_time - start_time
            @test elapsed_time < 10.0  # 10秒以内に完了
            @test length(output.t) > 0
        end
        
        @testset "Medium Sample Size" begin
            N = 1000
            start_time = time()
            output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
            end_time = time()
            
            elapsed_time = end_time - start_time
            @test elapsed_time < 30.0  # 30秒以内に完了
            @test length(output.t) > 0
        end
    end
    
    @testset "Scalability Tests" begin
        function U_test(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        # 次元のスケーラビリティ
        @testset "Dimension Scalability" begin
            dimensions = [1, 2, 5, 10]
            N = 500
            seed = 42
            
            for dim in dimensions
                @testset "Dimension $dim" begin
                    sampler = ZigZagAD(dim, U_test, grid_size=0)
                    xinit = zeros(dim)
                    vinit = ones(dim)
                    
                    start_time = time()
                    output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
                    end_time = time()
                    
                    elapsed_time = end_time - start_time
                    @test elapsed_time < 60.0  # 1分以内に完了
                    @test length(output.t) > 0
                end
            end
        end
        
        # サンプル数のスケーラビリティ
        @testset "Sample Size Scalability" begin
            dim = 2
            sample_sizes = [100, 500, 1000, 2000]
            seed = 42
            
            sampler = ZigZagAD(dim, U_test, grid_size=0)
            xinit = zeros(dim)
            vinit = ones(dim)
            
            for N in sample_sizes
                @testset "Sample Size $N" begin
                    start_time = time()
                    output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
                    end_time = time()
                    
                    elapsed_time = end_time - start_time
                    @test elapsed_time < 120.0  # 2分以内に完了
                    @test length(output.t) > 0
                end
            end
        end
    end
    
    @testset "Memory Usage Tests" begin
        function U_test(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 5
        sampler = ZigZagAD(dim, U_test, grid_size=0)
        xinit = zeros(dim)
        vinit = ones(dim)
        seed = 42
        
        # メモリ使用量のテスト
        @testset "Memory Usage" begin
            N = 1000
            start_memory = Base.gc_bytes()
            output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
            end_memory = Base.gc_bytes()
            
            memory_used = end_memory - start_memory
            @test memory_used < 100 * 1024 * 1024  # 100MB以内
            
            # ガベージコレクション後の確認
            GC.gc()
            @test length(output.t) > 0
        end
        
        # 大量のサンプリングでのメモリテスト
        @testset "Large Sample Memory" begin
            N = 10000
            start_memory = Base.gc_bytes()
            output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
            end_memory = Base.gc_bytes()
            
            memory_used = end_memory - start_memory
            @test memory_used < 500 * 1024 * 1024  # 500MB以内
            
            # サンプル生成でのメモリテスト
            start_memory = Base.gc_bytes()
            samples = sample_from_skeleton(sampler, N, output)
            end_memory = Base.gc_bytes()
            
            memory_used = end_memory - start_memory
            @test memory_used < 100 * 1024 * 1024  # 100MB以内
        end
    end
    
    @testset "AD Backend Performance" begin
        function U_test(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 3
        N = 500
        xinit = zeros(dim)
        vinit = ones(dim)
        seed = 42
        
        backends = ["ForwardDiff", "Zygote"]
        results = Dict()
        
        for backend in backends
            @testset "$backend Performance" begin
                sampler = ZigZagAD(dim, U_test, grid_size=0, AD_backend=backend)
                
                start_time = time()
                output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
                end_time = time()
                
                elapsed_time = end_time - start_time
                @test elapsed_time < 60.0  # 1分以内に完了
                @test length(output.t) > 0
                
                results[backend] = elapsed_time
            end
        end
        
        # バックエンド間の性能比較
        @testset "Backend Comparison" begin
            if haskey(results, "ForwardDiff") && haskey(results, "Zygote")
                # 性能差が極端でないことを確認
                ratio = results["Zygote"] / results["ForwardDiff"]
                @test 0.1 < ratio < 10.0  # 10倍以上の差がないことを確認
            end
        end
    end
    
    @testset "Grid Size Performance" begin
        function U_test(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 2
        N = 500
        xinit = zeros(dim)
        vinit = ones(dim)
        seed = 42
        
        grid_sizes = [0, 5, 10, 20]
        results = Dict()
        
        for grid_size in grid_sizes
            @testset "Grid Size $grid_size" begin
                sampler = ZigZagAD(dim, U_test, grid_size=grid_size)
                
                start_time = time()
                output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
                end_time = time()
                
                elapsed_time = end_time - start_time
                @test elapsed_time < 60.0  # 1分以内に完了
                @test length(output.t) > 0
                
                results[grid_size] = elapsed_time
            end
        end
        
        # グリッドサイズによる性能変化の確認
        @testset "Grid Size Impact" begin
            if haskey(results, 0) && haskey(results, 20)
                # グリッドサイズが大きくなっても極端に遅くならないことを確認
                ratio = results[20] / results[0]
                @test ratio < 5.0  # 5倍以上の差がないことを確認
            end
        end
    end
    
    @testset "Complex Potential Performance" begin
        # 複雑なポテンシャルでの性能テスト
        function U_banana(x::AbstractVector)
            mean_x2 = (x[1]^2 - 1)
            return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
        end
        
        function U_funnel(x::AbstractVector)
            d = length(x)
            v = x[1]
            return v^2 / 2 + (d-1) * log(v) + sum(x[2:end].^2) / (2 * v^2)
        end
        
        potentials = [
            ("Banana", U_banana, 3),
            ("Funnel", U_funnel, 3)
        ]
        
        N = 300
        seed = 42
        
        for (name, U, dim) in potentials
            @testset "$name Performance" begin
                sampler = ZigZagAD(dim, U, grid_size=0)
                xinit = zeros(dim)
                vinit = ones(dim)
                
                start_time = time()
                output = sample_skeleton(sampler, N, xinit, vinit, seed=seed)
                end_time = time()
                
                elapsed_time = end_time - start_time
                @test elapsed_time < 90.0  # 1.5分以内に完了
                @test length(output.t) > 0
            end
        end
    end
    
    @testset "Concurrent Execution" begin
        function U_test(x::Float64)
            return x^2 / 2
        end
        
        # 並列実行のテスト（基本的な確認）
        @testset "Parallel Safety" begin
            sampler = ZigZagAD(1, U_test, grid_size=0)
            N = 100
            seed = 42
            
            # 複数の実行を連続で行う
            results = []
            for i in 1:4
                start_time = time()
                output = sample_skeleton(sampler, N, 0.0, 1.0, seed=seed+i)
                end_time = time()
                
                elapsed_time = end_time - start_time
                @test elapsed_time < 30.0  # 30秒以内に完了
                @test length(output.t) > 0
                
                push!(results, output.t)
            end
            
            # 結果が異なることを確認
            for i in 1:length(results)
                for j in (i+1):length(results)
                    @test results[i] != results[j] || length(results[i]) == 0
                end
            end
        end
    end
end
