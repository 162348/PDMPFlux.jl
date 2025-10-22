# 自動微分バックエンドのテスト
using Test

@testset "AD Backends" begin
    
    @testset "ForwardDiff Backend" begin
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=grid_size, AD_backend="ForwardDiff")
        
        @test sampler.AD_backend == "ForwardDiff"
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
    end
    
    @testset "Zygote Backend" begin
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=grid_size, AD_backend="Zygote")
        
        @test sampler.AD_backend == "Zygote"
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
    end
    
    @testset "Enzyme Backend" begin
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=grid_size, AD_backend="Enzyme")
        
        @test sampler.AD_backend == "Enzyme"
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(output.x))
        @test all(isfinite.(output.v))
    end
    
    @testset "Gradient Consistency" begin
        function U_test(x::Vector{Float64})
            return x[1]^2 + 2*x[2]^2 + x[1]*x[2]
        end
        
        # 手動で計算した勾配
        function ∇U_manual(x::Vector{Float64})
            return [2*x[1] + x[2], 4*x[2] + x[1]]
        end
        
        # テスト点
        x_test = [1.0, 2.0]
        grad_manual = ∇U_manual(x_test)
        
        # ForwardDiffでの勾配
        import ForwardDiff
        grad_forward = ForwardDiff.gradient(U_test, x_test)
        
        # Zygoteでの勾配
        import Zygote
        grad_zygote = Zygote.gradient(U_test, x_test)[1]
        
        # 勾配の一致をテスト
        @test grad_manual ≈ grad_forward atol=1e-10
        @test grad_manual ≈ grad_zygote atol=1e-10
        @test grad_forward ≈ grad_zygote atol=1e-10
    end
end
