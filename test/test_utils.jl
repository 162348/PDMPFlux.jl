# ユーティリティ関数のテスト
using Test

@testset "Utility Functions" begin
    
    @testset "Potential Functions" begin
        # 1次元ガウシアン
        function U_Gauss_1D(x::Float64)
            return x^2 / 2
        end
        
        @test U_Gauss_1D(0.0) == 0.0
        @test U_Gauss_1D(1.0) == 0.5
        @test U_Gauss_1D(-2.0) == 2.0
        
        # 多次元ガウシアン
        function U_Gauss(x::AbstractVector)
            return sum(x.^2) / 2
        end
        
        @test U_Gauss([0.0, 0.0]) == 0.0
        @test U_Gauss([1.0, 1.0]) == 1.0
        @test U_Gauss([2.0, 0.0, 0.0]) == 2.0
        
        # コーシー分布
        function U_Cauchy(x::Float64)
            return log(1 + x^2)
        end
        
        @test U_Cauchy(0.0) == 0.0
        @test U_Cauchy(1.0) ≈ log(2)
    end
    
    @testset "Banana Function" begin
        function U_banana(x::AbstractVector)
            mean_x2 = (x[1]^2 - 1)
            return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
        end
        
        # 2次元バナナ関数のテスト
        x2d = [0.0, 1.0]
        result_2d = U_banana(x2d)
        @test isfinite(result_2d)
        
        # 3次元バナナ関数のテスト
        x3d = [0.0, 1.0, 0.0]
        result_3d = U_banana(x3d)
        @test isfinite(result_3d)
        @test result_2d == result_3d  # 3次元目は0なので同じ結果になるはず
    end
    
    @testset "Funnel Function" begin
        function U_funnel(x::AbstractVector)
            y = x[1]
            log_density_y = - y^2 / 6
            variance_other = exp(y/2)
            log_density_other = - sum(x[2:end].^2) / (2 * variance_other)
            return - log_density_y - log_density_other
        end
        
        # 2次元ファンネル関数のテスト
        x2d = [0.0, 1.0]
        result_2d = U_funnel(x2d)
        @test isfinite(result_2d)
        
        # 3次元ファンネル関数のテスト
        x3d = [0.0, 1.0, 0.0]
        result_3d = U_funnel(x3d)
        @test isfinite(result_3d)
    end
end
