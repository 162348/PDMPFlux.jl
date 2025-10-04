using Plots
using Random
using LaTeXStrings

# パラメータの設定
μ = 0.1  # ドリフト項
σ = 0.2  # ボラティリティ
T = 1.0  # 時間の終点
N = 5000 # 時間ステップ数
dt = T/N # 時間ステップ幅
S₀ = 1.0 # 初期値

# 幾何ブラウン運動のシミュレーション
function simulate_gbm(μ, σ, T, N, S₀)
    t = range(0, T, length=N)
    S = zeros(N)
    S[1] = S₀
    
    for i in 2:N
        dW = √dt * randn()
        S[i] = S[i-1] * exp((μ - 0.5σ^2) * dt + σ * dW)
    end
    
    return t, S
end

# パラメータの設定
σ = 0.2  # 拡散係数
T = 1.0  # 時間の終点
N = 5000 # 時間ステップ数
dt = T/N # 時間ステップ幅
S₀ = 1.0 # 初期値
# OU 過程のシミュレーション
function simulate_ou(σ, T, N, S₀)
    t = range(0, T, length=N)
    S = zeros(N)
    S[1] = S₀
    
    for i in 2:N
        dW = √dt * randn()
        S[i] = S[i-1] - σ * S[i-1] * dt / 2 + sqrt(σ) * dW
    end
    
    return t, S
end

# シミュレーションの実行
t, S = simulate_ou(σ, T, N, S₀)

# アニメーションの作成
anim = @animate for i in 1:10:N
    plot(t[1:i], S[1:i], 
        #  label="Geometric Brownian Motion",
         ylabel=L"z_t",
         linewidth=2,
         legend=false,
         background_color="#F0F1EB",
         linecolor="#E95420",
         size=(600, 600),
        #  xlim=(0, T),
        #  ylim=(0, 1.11),
    )
end

# アニメーションの保存
gif(anim, "OU_simulation.gif", fps=60)

