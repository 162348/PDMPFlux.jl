using Test
using Statistics
using LinearAlgebra
using Random

@testset "Quick Tests" begin
    
    @testset "Basic Functionality" begin
        # 基本的な機能のテスト（高速）
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        # 最小限のサンプリングテスト
        sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        output = sample_skeleton(sampler, 100, 0.0, 1.0, seed=42)
        
        @test length(output.t) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
        
        # サンプル生成
        samples = sample_from_skeleton(sampler, 100, output)
        @test size(samples) == (1, 100)
        @test all(isfinite.(samples))
    end
    
    @testset "All Sampler Types" begin
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        # 各サンプラーの基本動作確認
        samplers = [
            ("ZigZag", ZigZagAD(dim, U_Gauss_2D, grid_size=0)),
            ("BPS", BPSAD(dim, U_Gauss_2D, grid_size=0)),
            ("Boomerang", BoomerangAD(dim, U_Gauss_2D, grid_size=0)),
        ]
        
        # ForwardECMCは3次元以上が必要
        dim_3d = 3
        function U_Gauss_3D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        xinit_3d = [0.0, 0.0, 0.0]
        vinit_3d = [1.0, 1.0, 1.0]
        
        @testset "ForwardECMC" begin
            sampler = ForwardECMCAD(dim_3d, U_Gauss_3D, grid_size=10)
            output = sample_skeleton(sampler, 50, xinit_3d, vinit_3d, seed=seed)
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
        
        for (name, sampler) in samplers
            @testset "$name" begin
                output = sample_skeleton(sampler, 50, xinit, vinit, seed=seed)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
            end
        end
    end
    
    @testset "AD Backends" begin
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        backends = ["ForwardDiff", "Zygote"]
        
        for backend in backends
            @testset "$backend" begin
                sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=0, AD_backend=backend)
                output = sample_skeleton(sampler, 50, xinit, vinit, seed=seed)
                @test length(output.t) > 0
                @test all(isfinite.(output.t))
            end
        end
    end
    
    @testset "Error Handling" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        # 基本的なエラーハンドリング
        # 注意: 実際のエラーハンドリングは実装によって異なる場合があります
        @test_nowarn ZigZagAD(1, U_Gauss_1D, grid_size=0)
        @test_nowarn sample_skeleton(ZigZagAD(1, U_Gauss_1D, grid_size=0), 10, 0.0, 1.0, seed=42)
    end
    
    @testset "Reproducibility" begin
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        sampler = ZigZagAD(1, U_Gauss_1D, grid_size=0)
        seed = 12345
        
        output1 = sample_skeleton(sampler, 50, 0.0, 1.0, seed=seed)
        output2 = sample_skeleton(sampler, 50, 0.0, 1.0, seed=seed)
        
        @test output1.t ≈ output2.t
    end
end
