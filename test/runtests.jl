using Test
using PDMPFlux
using Random
using Distributions
using LinearAlgebra
using Statistics

# テスト用のシードを設定
Random.seed!(42)

# テスト設定
const TEST_TIMEOUT = 300  # 5分のタイムアウト
const QUICK_TEST_SAMPLES = 100
const STANDARD_TEST_SAMPLES = 500
const EXTENDED_TEST_SAMPLES = 1000

# 基本テスト（必須）
println("Running basic tests...")
include("test_utils.jl")
include("test_samplers.jl")
include("test_ad_backends.jl")

# 機能テスト
println("Running functionality tests...")
include("test_diagnostics.jl")
include("test_plotting.jl")

# 拡張テスト（時間がかかる）
println("Running extended tests...")
include("test_comprehensive.jl")
# include("test_error_handling.jl")
include("test_property_based.jl")
include("test_integration.jl")

# 新しいテスト
# println("Running coverage tests...")
# include("test_coverage.jl")
# include("test_stability.jl")
# include("test_performance.jl")

# テストヘルパー（必要に応じて）
# include("test_helpers.jl")

println("All tests completed successfully!")