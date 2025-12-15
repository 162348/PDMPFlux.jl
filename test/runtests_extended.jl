using Test
using PDMPFlux
using Random
using Distributions
using LinearAlgebra
using Statistics

# テスト用のシードを設定
Random.seed!(42)

# Quality assurance checks (Aqua.jl)
include("test_aqua.jl")

# 基本テスト
include("test_utils.jl")
include("test_samplers.jl")
include("test_ad_backends.jl")
include("test_diagnostics.jl")
include("test_plotting.jl")

# 拡張テスト
include("test_comprehensive.jl")
include("test_error_handling.jl")
include("test_property_based.jl")
include("test_integration.jl")

println("Extended tests completed successfully!")
