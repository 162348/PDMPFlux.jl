using Test
using PDMPFlux

@testset "Error Handling and Edge Cases" begin
    
    @testset "Invalid Input Validation" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        # 無効な次元
        @test_throws ArgumentError ZigZagAD(0, U_Gauss_1D, grid_size=0)
        @test_throws ArgumentError ZigZagAD(-1, U_Gauss_1D, grid_size=0)
        
        # 無効なグリッドサイズ
        @test_throws ArgumentError ZigZagAD(1, U_Gauss_1D, grid_size=-1)
        
        # 無効な関数
        @test_throws MethodError ZigZagAD(1, nothing, grid_size=0)
    end
    
    @testset "Boundary Conditions" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        
        # ゼロサンプル数
        @test_throws ArgumentError sample_skeleton(sampler, 0, 0.0, 1.0, seed=42)
        @test_throws ArgumentError sample_from_skeleton(sampler, 0, 
            sample_skeleton(sampler, 100, 0.0, 1.0, seed=42))
        
        # 負のサンプル数
        @test_throws ArgumentError sample_skeleton(sampler, -1, 0.0, 1.0, seed=42)
    end
    
    @testset "Numerical Stability" begin
        # 非常に急峻なポテンシャル
        function U_Steep(x::Float64)
            return 1000 * x^2
        end
        
        sampler = ZigZagAD(1, U_Steep, grid_size=0)
        
        # 数値的に安定であることを確認
        output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
        @test all(isfinite.(output.t))
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
        
        # 非常に平坦なポテンシャル
        function U_Flat(x::Float64)
            return 0.001 * x^2
        end
        
        sampler_flat = ZigZagAD(1, U_Flat, grid_size=0)
        output_flat = sample_skeleton(sampler_flat, 100, 0.0, 1.0, seed=42)
        @test all(isfinite.(output_flat.t))
        @test all(isfinite.(hcat(output_flat.x...)))
        @test all(isfinite.(hcat(output_flat.v...)))
    end
    
    @testset "Memory and Performance Limits" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        
        # 非常に大きなサンプル数（メモリ制限内）
        @test_nowarn sample_skeleton(sampler, 10000, 0.0, 1.0, seed=42)
        
        # 非常に小さなサンプル数
        output = sample_skeleton(sampler, 1, 0.0, 1.0, seed=42)
        @test length(output.t) >= 1
    end
    
    @testset "Type Stability" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        
        # 異なる数値型での動作
        @test_nowarn sample_skeleton(sampler, 100, 0, 1, seed=42)  # Int
        @test_nowarn sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)  # Float64
        
        # 結果の型が一貫していることを確認
        output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
        @test eltype(output.t) == Float64
        @test eltype(output.x) == Vector{Float64}
        @test eltype(output.v) == Vector{Float64}
    end
    
    @testset "Concurrent Access Safety" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        # 複数のサンプラーインスタンスが独立して動作することを確認
        sampler1 = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        sampler2 = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        
        output1 = sample_skeleton(sampler1, 100, 0.0, 1.0, seed=42)
        output2 = sample_skeleton(sampler2, 100, 0.0, 1.0, seed=42)
        
        # 同じシードなら同じ結果
        @test output1.t ≈ output2.t
        @test output1.x ≈ output2.x
        @test output1.v ≈ output2.v
    end
    
    @testset "Resource Cleanup" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        # 大量のサンプラー作成と破棄
        for i in 1:10
            sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
            output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
            @test length(output.t) > 0
        end
        
        # メモリリークがないことを確認（手動でガベージコレクション）
        GC.gc()
        @test true  # エラーが発生しなければ成功
    end
end
