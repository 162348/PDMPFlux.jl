# PDMPFlux テストガイド

このディレクトリには、PDMPFluxパッケージの包括的なテストスイートが含まれています。

## テスト構造

```
test/
├── runtests.jl              # メインテストファイル（全テスト）
├── runtests_quick.jl        # クイックテスト（CI/CD用）
├── runtests_extended.jl     # 拡張テスト（包括的）
├── test_config.jl           # テスト設定とユーティリティ
├── test_utils.jl            # ユーティリティ関数のテスト
├── test_samplers.jl         # サンプラーのテスト
├── test_ad_backends.jl      # 自動微分バックエンドのテスト
├── test_diagnostics.jl      # 診断機能のテスト
├── test_plotting.jl         # プロット機能のテスト
├── test_comprehensive.jl    # 包括的テスト
├── test_error_handling.jl   # エラーハンドリングテスト
├── test_property_based.jl   # プロパティベーステスト
├── test_integration.jl      # 統合テスト
├── test_quick.jl            # クイックテスト
├── test_coverage.jl         # カバレッジ向上テスト
├── test_stability.jl        # 数値的安定性テスト
├── test_performance.jl      # パフォーマンステスト
├── test_helpers.jl          # テスト用ヘルパー関数
├── benchmarks.jl            # パフォーマンスベンチマーク
└── README.md               # このファイル
```

## テストの実行方法

### 1. クイックテスト（推奨：CI/CD用）

```julia
using Pkg
Pkg.test("PDMPFlux", test_args=["test/runtests_quick.jl"])
```

または：

```bash
julia --project=. test/runtests_quick.jl
```

### 2. 標準テスト（開発用）

```julia
using Pkg
Pkg.test("PDMPFlux")
```

または：

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

### 3. 拡張テスト（包括的）

```bash
julia --project=. test/runtests_extended.jl
```

### 4. 個別のテストファイルの実行

```julia
using Test
include("test/test_utils.jl")
include("test/test_samplers.jl")
# など
```

### 5. パフォーマンスベンチマークの実行

```julia
include("test/benchmarks.jl")
run_benchmarks()
```

## テストの種類

### 1. ユニットテスト (`test_utils.jl`)
- ポテンシャル関数の基本的な動作テスト
- 数学的な関数の正確性テスト

### 2. サンプラーテスト (`test_samplers.jl`)
- 各サンプラー（ZigZag、ForwardECMC、BPS等）の基本動作テスト
- スケルトンサンプリングとサンプル生成のテスト
- 統計的妥当性の検証

### 3. 自動微分バックエンドテスト (`test_ad_backends.jl`)
- ForwardDiff、Zygote、Enzymeの動作テスト
- 勾配計算の一貫性テスト

### 4. 診断テスト (`test_diagnostics.jl`)
- 診断機能の動作テスト
- サンプル統計量の妥当性テスト

### 5. プロットテスト (`test_plotting.jl`)
- プロット関数の動作テスト
- アニメーション生成のテスト

### 6. 包括的テスト (`test_comprehensive.jl`)
- 全サンプラータイプの統合テスト
- エッジケースと境界条件のテスト
- 再現性とパフォーマンスのテスト

### 7. エラーハンドリングテスト (`test_error_handling.jl`)
- 無効な入力の検証
- 数値的安定性のテスト
- 型安定性の確認

### 8. プロパティベーステスト (`test_property_based.jl`)
- 数学的不変量の検証
- 統計的性質のテスト
- 物理的性質の確認

### 9. 統合テスト (`test_integration.jl`)
- 実際の使用例とワークフローのテスト
- サンプラー間の比較
- 診断・可視化パイプラインのテスト

### 10. クイックテスト (`test_quick.jl`)
- CI/CD用の高速テスト
- 基本的な機能の確認
- 最小限のサンプリングテスト

### 11. カバレッジテスト (`test_coverage.jl`)
- エッジケースと境界条件のテスト
- 高次元での動作確認
- 複雑なポテンシャル関数のテスト
- サンプラー間の比較テスト

### 12. 安定性テスト (`test_stability.jl`)
- 数値的安定性のテスト
- 極値での動作確認
- 特異点での動作確認
- メモリ管理のテスト

### 13. テスト設定 (`test_config.jl`)
- テスト用の設定とユーティリティ
- 共通のポテンシャル関数
- テスト用のアサーション関数

### 14. パフォーマンステスト (`test_performance.jl`)
- スケーラビリティテスト
- メモリ使用量テスト
- ADバックエンド性能比較
- 並列実行テスト

### 15. テストヘルパー (`test_helpers.jl`)
- テスト用のマクロと関数
- 共通のテストパターン
- パフォーマンス測定ユーティリティ

## テストの追加方法

新しいテストを追加する場合は、以下のガイドラインに従ってください：

1. **適切なテストセットを使用**: `@testset`を使用してテストをグループ化
2. **明確なテスト名**: テストの目的が分かる名前を付ける
3. **エラーハンドリング**: `@test_nowarn`や`@test_throws`を適切に使用
4. **数値テスト**: `≈`（近似等価）を使用して浮動小数点の比較を行う
5. **テストデータ**: 再現可能な結果のためにシードを設定

### 例：

```julia
@testset "My New Feature" begin
    # テストのセットアップ
    x = [1.0, 2.0, 3.0]
    
    # 基本的なテスト
    @test length(x) == 3
    
    # 近似テスト
    @test sum(x) ≈ 6.0 atol=1e-10
    
    # エラーテスト
    @test_throws ArgumentError some_function(-1)
    
    # 警告なしテスト
    @test_nowarn some_plotting_function(x)
end
```

## 継続的インテグレーション

このパッケージはGitHub Actionsでの自動テストをサポートしています。テストは以下の環境で実行されます：

- Julia 1.11.1
- 複数のOS（Linux、macOS、Windows）

## トラブルシューティング

### テストが失敗する場合

1. **依存関係の確認**: 必要なパッケージがインストールされているか確認
2. **シードの確認**: ランダム性に依存するテストでは、シードが適切に設定されているか確認
3. **数値精度**: 浮動小数点の比較では適切な許容誤差を設定

### パフォーマンステストが遅い場合

- `benchmarks.jl`のサンプル数を減らす
- より小さな次元でテストする
- 特定のバックエンドのみをテストする

## 貢献

新しいテストを追加する際は、以下の点を考慮してください：

1. テストは独立して実行できること
2. テストは再現可能であること
3. テストは適切なエラーメッセージを提供すること
4. ドキュメントを更新すること
