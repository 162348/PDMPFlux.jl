using Test
using PDMPFlux
using Random
using Distributions
using LinearAlgebra
using Statistics

# テスト用のシードを設定
Random.seed!(42)

# クイックテスト設定
const QUICK_TEST_SAMPLES = 50
const QUICK_TEST_TIMEOUT = 60  # 1分のタイムアウト

println("Running quick tests for CI/CD...")

# クイックテストのみを実行
include("test_quick.jl")

# 基本的なエラーハンドリングテスト
println("Running basic error handling tests...")
include("test_error_handling.jl")

println("Quick tests completed successfully!")
