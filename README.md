# Restaurant Demand Forecasting

ファミリーレストランチェーン10店舗の売上予測コンペティション用リポジトリ。過去の売上・来店・注文履歴や施策情報を使い、2025年の **日付 × 店舗 × 商品** ごとの売上金額を予測する。

## 課題概要

目的変数の `sales_amount` は、クーポンなどによる値引き後の売上金額（円）。2024年末の時点で、2025年の1年分をまとめて予測する。

| 区分 | 期間 | 日数 | 店舗数 | 商品数 | 行数 |
|---|---|---:|---:|---:|---:|
| 学習データ | 2022-01-01〜2024-12-31 | 1,096 | 10 | 84 | 920,640 |
| 予測対象 | 2025-01-01〜2025-12-31 | 365 | 10 | 84 | 306,600 |

`test.csv` の全行に予測値を入力し、配布ツールで6つの集計層を含む提出形式に変換する。データの詳細は `input/data_doc.pdf`、提出形式と評価重みの詳細は `input/submission_scripts.zip` 内のガイドを参照。

## セットアップ

Python 3.12 と [uv](https://docs.astral.sh/uv/) を使用する。依存関係は `uv.lock` で管理し、プロジェクト内の `.venv/` にインストールする。

```bash
uv sync
```

配布データは `input/` に配置する。生データ、ログ、加工済みデータ、モデル、実行結果は Git 管理しない。EDA ノートブックは `logs.zip` を展開せずに読み込める。

## データ

| ファイル | 内容 |
|---|---|
| `train.csv` | 2022〜2024年の日付 × 店舗 × 商品ごとの売上実績 |
| `test.csv` | 2025年の予測対象。`sales_amount` の初期値は0 |
| `stores.csv` | 店舗の立地、床面積、開店日、営業時間 |
| `seat_layout.csv` | 店舗ごとの卓構成・総席数の履歴と2025年の変更計画 |
| `menus.csv` | 商品名、カテゴリ、販売時間帯、定番・季節限定・コラボなどの属性 |
| `menu_catalog_history.csv` | 商品の販売期間と関連する企画ID。2025年の予定を含む |
| `price_history.csv` | 店舗・商品別の割引前価格と適用期間。2025年の価格改定計画を含む |
| `campaign_calendar.csv` | 日別の企画、割引率、対象商品群・店舗など。2025年の予定を含む |
| `calendar.csv` | 祝日、2024年までの天候実績、2025年までの天候平年値 |
| `logs.zip` | 2022〜2024年の店舗別来店ログ・注文明細（Parquet） |
| `submission_scripts.zip` | 日次予測の変換・検証ツール、提出サンプル、提出ガイド |
| `data_doc.pdf` | 課題・データ・評価方法の説明 |

### 予測時点で利用できる情報

2025年については、価格、キャンペーン、販売期間、席構成、祝日、天候の平年値が予定・計画として配布されている。各ファイルの適用期間と対象店舗を確認して使う。

2025年の売上、来店・注文ログ、天候実績は利用できない。`weather_observed`、`temperature_observed`、`precipitation_observed` は2024年までの実績であり、2025年には空欄となる。平年値は季節的な目安で、特定日の天気予報ではない。特徴量候補と利用可能時点は [eda/FEATURES.md](eda/FEATURES.md) に整理している。

## 構成

```text
input/                 配布データ（Git管理外）
data/processed/        加工済みデータ（Git管理外）
data/external/         外部データ（Git管理外）
eda/                   探索・可視化ノートブックと分析結果
notebooks/             その他の探索・可視化
src/
  config.py             パスと設定
  data.py               生データの読み込み
  availability.py       販売期間・参加店舗による販売可能性
  baselines.py          過去平均による基準予測
  adjusted_baseline.py  価格・祝日・店舗成長を考慮した平均予測
  forecast.py           天候・季節・席構成を考慮した平均予測
  metrics.py            6層集計と時系列検証スコア
  runner.py             時系列分割、予測、評価、成果物保存
  submission.py         配布された公式変換・検証スクリプトの呼び出し
exp/expNNN_説明/       実験ごとのコードと記録
models/                学習済みモデル（Git管理外）
output/runs/           実行時コード、設定、検証結果、日次予測（Git管理外）
output/submissions/    提出形式のファイル（Git管理外）
output/figures/        図表（Git管理外）
```

## EDA

```bash
uv run jupyter lab
```

実行済みノートブックの一覧と主な発見は [eda/README.md](eda/README.md) を参照。売上、マスタ、カレンダー、全店舗の来店・注文ログ、提出サンプルを分析している。

## 実験

```bash
# 店舗・商品・曜日ごとの過去平均
uv run python -m exp.exp001_baseline.run

# 販売期間・コラボ参加店舗を考慮した曜日別平均
uv run python -m exp.exp002_availability_weekday.run

# 共通処理を使い exp002 を再現
uv run python -m exp.exp003_shared_baseline.run
```

各実験は2022〜2023年で2024年を検証し、全学習期間を使って2025年を予測する。実行結果は `output/runs/<実行名>/` に保存される。`predictions.csv` は提出形式に変換する前の日次予測。

| 実験 | 方法 | 2024年の6層正規化 RMSE |
|---|---|---:|
| [exp001](exp/exp001_baseline/README.md) | 店舗・商品・曜日別の過去平均 | 1.198543 |
| [exp002](exp/exp002_availability_weekday/README.md) | 販売可能性を考慮した曜日別平均 | 0.828145 |
| [exp003](exp/exp003_shared_baseline/README.md) | 共通処理による exp002 の再現 | 0.828145 |
| [exp004](exp/exp004_recent_weight/README.md) | 直近年の重み付け | 0.757162 |
| [exp005](exp/exp005_holiday_factor/README.md) | 祝日倍率 | 0.741816 |
| [exp006](exp/exp006_price_normalization/README.md) | 価格による正規化 | 0.626834 |
| [exp007](exp/exp007_month_factor/README.md) | 商品別の月別倍率（不採用） | 0.632723 |
| [exp008](exp/exp008_store_growth/README.md) | 店舗別の前年成長率 | 0.593427 |
| [exp009](exp/exp009_category_holiday/README.md) | カテゴリ別の祝日倍率 | 0.593219 |
| [exp010](exp/exp010_monthly_means/README.md) | 月別平均（不採用） | 0.641097 |
| [exp011](exp/exp011_layout_means/README.md) | 席レイアウト別平均（不採用） | 0.763858 |
| [exp012](exp/exp012_shared_adjusted/README.md) | 共通処理による exp009 の再現 | 0.593219 |
| [exp013](exp/exp013_price_elasticity/README.md) | 価格上昇による数量減の仮定 | 0.562180 |
| [exp014](exp/exp014_store_month_factor/README.md) | 店舗別の月次倍率（不採用） | 0.591725 |
| [exp015](exp/exp015_coupon_context/README.md) | クーポン予定の条件付き平均 | 0.561457 |
| [exp016](exp/exp016_recent_growth/README.md) | 直近四半期の店舗成長率（不採用） | 0.592629 |
| [exp017](exp/exp017_near_year_mean/README.md) | 前年近傍の同曜日平均（不採用） | 0.570941 |
| [exp018](exp/exp018_normal_rain/README.md) | 降水確率平年値による条件付き平均 | 0.560128 |
| [exp019](exp/exp019_latest_layout_level/README.md) | 最新レイアウトの店舗水準 | 0.432661 |
| [exp020](exp/exp020_layout_month_factor/README.md) | 最新レイアウトと月次倍率（不採用） | 0.461224 |
| [exp021](exp/exp021_layout_category/README.md) | 最新レイアウトの店舗・カテゴリ別水準 | 0.429323 |
| [exp022](exp/exp022_partial_growth/README.md) | 最新水準への部分的な年次成長（不採用） | 0.471122 |
| [exp023](exp/exp023_layout_item/README.md) | 最新レイアウトの商品別水準（不採用） | 0.432513 |
| [exp024](exp/exp024_adjusted_month/README.md) | レイアウト水準を除いた月次倍率 | 0.401270 |
| [exp025](exp/exp025_nonholiday/README.md) | 非祝日の条件付き平均 | 0.396673 |
| [exp026](exp/exp026_category_season/README.md) | カテゴリ別の月次倍率（不採用） | 0.396867 |
| [exp027](exp/exp027_near_holiday/README.md) | 祝日前後の非祝日（不採用） | 0.397538 |
| [exp028](exp/exp028_holiday_date/README.md) | 祝日ごとの条件付き平均（不採用） | 0.399902 |
| [exp029](exp/exp029_dayofyear_smooth/README.md) | 日付近傍の平滑化平均 | 0.393254 |
| [exp030](exp/exp030_store_dayofyear/README.md) | 店舗別の日付近傍平均 | 0.393119 |
| [exp031](exp/exp031_adjusted_weather/README.md) | レイアウト・月別水準を除いた天候別平均 | 0.392608 |
| [exp032](exp/exp032_adjusted_holiday/README.md) | レイアウト水準を除いた祝日倍率 | 0.390239 |
| [exp033](exp/exp033_adjusted_coupon/README.md) | レイアウト水準を除いた施策倍率（不採用） | 0.390396 |
| [exp034](exp/exp034_unit_elasticity/README.md) | 価格上昇と数量減が相殺する仮定（不採用） | 0.430591 |
| [exp035](exp/exp035_partial_item_layout/README.md) | 商品別レイアウト倍率の部分的反映（不採用） | 0.391715 |
| [exp038](exp/exp038_month_nonholiday/README.md) | 祝日水準を除いた月次倍率（不採用） | 0.390898 |
| [exp039](exp/exp039_seated_level/README.md) | 着席組数の最新レイアウト水準（不採用） | 0.424181 |
| [exp040](exp/exp040_shared_forecast/README.md) | 共通処理による exp032 の再現 | 0.390239 |
| [exp041](exp/exp041_planned_layout/README.md) | 予定されたレイアウト変更（不採用） | 0.466053 |
| [exp042](exp/exp042_matched_layout_growth/README.md) | 同一レイアウトで比較した年次変化 | 0.384175 |
| [exp043](exp/exp043_full_matched_growth/README.md) | 同一年次変化の全量継続（不採用） | 0.388937 |
| [exp044](exp/exp044_category_growth/README.md) | カテゴリ別の同一条件での年次変化（不採用） | 0.384599 |
| [exp045](exp/exp045_nonholiday_growth/README.md) | 非祝日に限定した同一条件の年次変化 | 0.383407 |
| [exp046](exp/exp046_recent_matched_growth/README.md) | 直近四半期の同一条件の年次変化 | 0.376520 |
| [exp047](exp/exp047_horizon_growth/README.md) | 予測期間の経過に応じた年次変化 | 0.374720 |
| [exp048](exp/exp048_paired_growth/README.md) | 364日前を対応付けた年次変化 | 0.374623 |
| [exp054](exp/exp054_rolling_quarter/README.md) | 基準日に合わせた直近3か月 | 0.374623 |
| [exp056](exp/exp056_cross_layout_growth/README.md) | レイアウト水準を調整した前年ペア（不採用） | 0.382160 |
| [exp057](exp/exp057_constant_quantity/README.md) | 定価変更後の数量一定の仮定（不採用） | 0.404467 |
| [exp058](exp/exp058_weekday_growth/README.md) | 曜日別の直近年次変化（不採用） | 0.376655 |
| [exp059](exp/exp059_recent_level/README.md) | 重み付き履歴平均から直近水準への補正 | 0.374501 |
| [exp060](exp/exp060_fallback_level/README.md) | 直近水準補正の年間比較への補完 | 0.372027 |
| [exp062](exp/exp062_eligible_level/README.md) | 販売可能日数に応じた直近水準補正（不採用） | 0.372068 |
| [exp064](exp/exp064_season_adjusted_layout/README.md) | 月別傾向を除いた最新レイアウト水準 | 0.371475 |
| [exp066](exp/exp066_robust_history/README.md) | 極端値候補の上限化（不採用） | 0.371478 |
| [exp049](exp/exp049_category_paired_growth/README.md) | カテゴリ別の364日ペア年次変化（不採用） | 0.374727 |
| [exp050](exp/exp050_weighted_paired_growth/README.md) | 直近四半期を重視した通年ペア（不採用） | 0.385467 |
| [exp051](exp/exp051_arrival_growth/README.md) | 来店組数の同一条件の年次変化（不採用） | 0.376211 |
| [exp052](exp/exp052_weather_adjusted_growth/README.md) | 過去の天候構成を調整した前年比（不採用） | 0.375012 |

2023年6月末を基準日とした別期間の参考検証は [exp053](exp/exp053_2023_backtest/README.md)、[exp055](exp/exp055_rolling_backtest/README.md)、[exp061](exp/exp061_level_backtest/README.md)、[exp063](exp/exp063_eligible_backtest/README.md)、[exp065](exp/exp065_layout_backtest/README.md) を参照。2022年内の半年予測誤差を尺度とするため、通年スコアとは比較できない。

参考として2024年6月末を基準日にした後半6か月の検証では、[exp036](exp/exp036_halfyear_032/README.md) が0.332109、[exp037](exp/exp037_halfyear_025/README.md) が0.340059。検証期間が上記の通年検証と重なり、予測期間も異なるため、独立した検証や通年スコアとは扱わない。

スコアは検証用に再計算した尺度による**ローカル参考値**で、2025年の公式スコアではない。exp002・exp003 の検証には、2024年の販売・企画予定が2023年末に既知だったという仮定がある。exp001 の6層スコアの算出方法も含め、検証方法は [eda/07_validation.ipynb](eda/07_validation.ipynb) を参照。

### 実験の進め方

- 実験ごとに `exp/expNNN_説明/` を作り、`run.py` と目的・方法・結果を記録する `README.md` を置く。結果を出した実験は変更せず、新しい番号で派生させる。
- `src.runner.run()` は履歴・検証・テストの分割、6層評価、予測、成果物保存を共通化する。検証・テストの前処理と予測には目的変数を渡さず、キーと行順を確認する。使い方は [exp003](exp/exp003_shared_baseline/run.py) を参照。
- 共通ランナーは実行時の設定、検証結果、日次予測、実験と `src/` のコード、依存関係、`input/` 直下のファイルと `src/` 内の Python ファイルのハッシュ、Git 状態を `output/runs/` に保存する。これらの入力・Python コード・依存関係設定の実行中の変更も検査する。
- 販売可能性は `src.availability`、基準予測は `src.baselines`、ローカルの6層評価は `src.metrics` にある。実行済みの exp001・exp002 は当時のコードのまま保存している。
- 予測期間に沿って時系列で検証し、各検証時点で利用できない情報を使わない。

## 評価と検証

評価は、日次予測を次の6層に集計し、それぞれの値に重みを掛けたうえで RMSE を計算する。値が小さいほど良い。月次は暦月の合計。

| 集計層 | 提出 `index` の先頭 | 重み W |
|---|---|---:|
| 日 × 店舗 × 商品 | `date_store_item` | 6.5892128296488259e-05 |
| 日 × 店舗 | `daily_store` | 3.1810727684176393e-05 |
| 日 × 商品（全店舗合計） | `daily_item` | 2.4883547837341564e-05 |
| 月 × 店舗 × 商品 | `monthly_store_item` | 1.366069112652441e-05 |
| 月 × 店舗 | `monthly_store` | 7.2589966574310893e-06 |
| 月 × 商品（全店舗合計） | `monthly_item` | 4.7801007736468625e-06 |

重みは対象期間全体で固定され、正解側にも同じ重みが掛かる。日次行単位の RMSE だけでは公式評価と一致しない。

配布された重みには2024年の実績に基づく尺度が含まれる。そのため2024年を検証する際は、検証年より前のデータで尺度と重みを再計算する。具体例は [eda/07_validation.ipynb](eda/07_validation.ipynb) を参照。

## 提出ファイルの作成

`predictions.csv` は `test.csv` と同じ `date,store_id,item_id,sales_amount` の4列・306,600行とし、予測値を円単位で入力する。ゼロの行も省略せず、負値・欠損値・無限大を含めない。

`src.submission` は配布 ZIP から公式スクリプトを一時展開して実行する。`RUN_NAME` は実行時に表示されたディレクトリ名に変更する。

```bash
RUN_NAME="実行時に表示されたディレクトリ名"

uv run python -m src.submission \
  --predictions "output/runs/${RUN_NAME}/predictions.csv" \
  --output "output/submissions/${RUN_NAME}.csv"

uv run python -m src.submission \
  --predictions "output/submissions/${RUN_NAME}.csv" --validate
```

提出するのは変換後の **ヘッダーなし2列・352,118行** のファイル。日×店舗×商品の予測から他の5層を同じツールで集計する。変換後は円単位ではないため、重みを再度掛けたり値を丸めたりしない。提出サンプルと同じ `index` と行順を維持する。

暫定評価と最終評価は非公開の異なるデータ部分を使い、同じ重みを適用する。参加者は区分を指定せず1ファイルを提出する。最終順位でスコアが同値の場合は応募日時の早い参加者が上位となり、入賞候補者には終了後にソースコード提出が求められる。詳細な条件は配布資料を確認する。
