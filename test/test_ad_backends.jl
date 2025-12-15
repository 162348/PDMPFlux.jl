# 自動微分バックエンドのテスト
using Test
using PDMPFlux

@testset "AD Backends" begin
    
    @testset "ForwardDiff Backend" begin
        function U_Gauss_2D(x::AbstractVector)
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
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
    end
    
    @testset "Zygote Backend" begin
        function U_Gauss_2D(x::AbstractVector)
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
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
    end
    
    @testset "ReverseDiff Backend" begin
        function U_Gauss_2D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=grid_size, AD_backend="ReverseDiff")
        
        @test sampler.AD_backend == "ReverseDiff"
        
        # スケルトンサンプリングのテスト
        N_sk = 1000
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        @test length(output.t) > 0
        @test all(isfinite.(output.t))
        @test all(isfinite.(hcat(output.x...)))
        @test all(isfinite.(hcat(output.v...)))
    end
    
    @testset "Enzyme Backend" begin
        # Enzymeが利用可能かチェック
        has_enzyme = false
        try
            import Enzyme
            has_enzyme = true
        catch
            has_enzyme = false
        end
        
        if has_enzyme
            function U_Gauss_2D(x::AbstractVector)
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
            @test all(isfinite.(hcat(output.x...)))
            @test all(isfinite.(hcat(output.v...)))
        else
            @warn "Enzyme package not available, skipping Enzyme backend tests"
        end
    end
    
    @testset "BPSAD with different backends" begin
        function U_Gauss_3D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        dim = 3
        grid_size = 0
        
        for backend in ["ForwardDiff", "Zygote", "ReverseDiff"]
            sampler = BPSAD(dim, U_Gauss_3D, grid_size=grid_size, AD_backend=backend)
            @test sampler.AD_backend == backend
            
            # 短いスケルトンサンプリングのテスト
            N_sk = 100
            xinit = randn(dim)
            vinit = randn(dim)
            vinit = vinit ./ norm(vinit)
            seed = 42
            
            output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
            
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
    end
    
    @testset "ForwardECMCAD with different backends" begin
        function U_Gauss_3D(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        # ForwardEventChain は dim >= 3 が必要
        dim = 3
        grid_size = 10
        
        for backend in ["ForwardDiff", "Zygote", "ReverseDiff"]
            sampler = ForwardECMCAD(dim, U_Gauss_3D, grid_size=grid_size, AD_backend=backend)
            
            # 短いスケルトンサンプリングのテスト
            N_sk = 100
            xinit = randn(dim)
            vinit = randn(dim)
            vinit = vinit ./ norm(vinit)
            seed = 42
            
            output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
            
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
    end
    
    @testset "Gradient Consistency" begin
        function U_test(x::AbstractVector)
            return x[1]^2 + 2*x[2]^2 + x[1]*x[2]
        end
        
        # 手動で計算した勾配
        function ∇U_manual(x::AbstractVector)
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
        
        # ReverseDiffでの勾配
        import ReverseDiff
        grad_reverse = ReverseDiff.gradient(U_test, x_test)
        
        # 勾配の一致をテスト
        @test grad_manual ≈ grad_forward atol=1e-10
        @test grad_manual ≈ grad_zygote atol=1e-10
        @test grad_manual ≈ grad_reverse atol=1e-10
        @test grad_forward ≈ grad_zygote atol=1e-10
        @test grad_forward ≈ grad_reverse atol=1e-10
        @test grad_zygote ≈ grad_reverse atol=1e-10
        
        # Enzymeでの勾配（利用可能な場合）
        try
            import Enzyme
            grad_enzyme = Enzyme.gradient(Enzyme.Reverse, U_test, x_test)[1]
            @test grad_manual ≈ grad_enzyme atol=1e-10
            @test grad_forward ≈ grad_enzyme atol=1e-10
        catch
            # Enzyme not available, skip
        end
    end
    
    @testset "1D function support" begin
        function U_1D(x::AbstractVector)
            return x[1]^2 / 2
        end
        
        dim = 1
        grid_size = 0
        
        for backend in ["ForwardDiff", "Zygote", "ReverseDiff"]
            sampler = ZigZagAD(dim, U_1D, grid_size=grid_size, AD_backend=backend)
            @test sampler.AD_backend == backend
            
            # 短いテスト
            N_sk = 50
            xinit = [0.0]
            vinit = [1.0]
            seed = 42
            
            output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
            
            @test length(output.t) > 0
            @test all(isfinite.(output.t))
        end
    end
end
