"""
プロット機能のテスト
"""
@testset "Plotting Functions" begin
    
    @testset "Basic Plotting Tests" begin
        # 1次元ガウシアンでのテスト
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        dim = 1
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_1D, grid_size=grid_size)
        
        N_sk = 1000
        xinit = 0.0
        vinit = 1.0
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        # プロット関数がエラーなく実行されることをテスト
        @test_nowarn plot_traj(output, 100)
        
        # アニメーション関数がエラーなく実行されることをテスト
        # 実際のアニメーション生成は時間がかかるので、短いバージョンでテスト
        @test_nowarn anim_traj(output, 10, filename="test_traj.gif")
        
        # テスト用ファイルを削除
        if isfile("test_traj.gif")
            rm("test_traj.gif")
        end
    end
    
    @testset "Joint Plot Tests" begin
        # 2次元ガウシアンでのテスト
        function U_Gauss_2D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 2
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_2D, grid_size=grid_size)
        
        N_sk = 1000
        N = 1000
        xinit = [0.0, 0.0]
        vinit = [1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        samples = sample_from_skeleton(sampler, N, output)
        
        # ジョイントプロットがエラーなく実行されることをテスト
        @test_nowarn jointplot(samples)
        
        # 座標指定でのジョイントプロット
        @test_nowarn jointplot(samples, coordinate_numbers=[1, 2])
    end
    
    @testset "3D Plotting Tests" begin
        # 3次元ガウシアンでのテスト
        function U_Gauss_3D(x::Vector{Float64})
            return sum(x.^2) / 2
        end
        
        dim = 3
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_3D, grid_size=grid_size)
        
        N_sk = 1000
        xinit = [0.0, 0.0, 0.0]
        vinit = [1.0, 1.0, 1.0]
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        # 3Dプロットがエラーなく実行されることをテスト
        @test_nowarn plot_traj(output, 100, plot_type="3D")
        
        # 3Dアニメーションがエラーなく実行されることをテスト
        @test_nowarn anim_traj(output, 10, plot_type="3D", filename="test_3d.gif")
        
        # テスト用ファイルを削除
        if isfile("test_3d.gif")
            rm("test_3d.gif")
        end
    end
    
    @testset "Plot Parameters" begin
        # プロットパラメータのテスト
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        dim = 1
        grid_size = 0
        sampler = ZigZagAD(dim, U_Gauss_1D, grid_size=grid_size)
        
        N_sk = 1000
        xinit = 0.0
        vinit = 1.0
        seed = 42
        
        output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed)
        
        # 異なるパラメータでのプロットテスト
        @test_nowarn plot_traj(output, 100, linewidth=2)
        @test_nowarn plot_traj(output, 100, background="white")
        
        # アニメーションパラメータのテスト
        @test_nowarn anim_traj(output, 10, fps=30, filename="test_params.gif")
        
        # テスト用ファイルを削除
        if isfile("test_params.gif")
            rm("test_params.gif")
        end
    end
end
