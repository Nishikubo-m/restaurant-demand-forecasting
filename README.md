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
