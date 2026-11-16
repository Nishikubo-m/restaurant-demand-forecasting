# 予測に向けた特徴量の候補と利用可能時点

2025年を予測する基準日は **2024年末**。ここに挙げるのは EDA に基づく候補であり、有効性や因果関係を保証しない。2024年を通年検証する場合は基準日を2023年末へ移し、売上・ログから作る統計をすべてその時点以前の情報だけで計算する。

| 情報・候補 | 元データと使い方 | 予測時点・検証時の注意 | 関連EDA |
|---|---|---|---|
| 日付・曜日・月・祝日 | `test.csv`, `calendar.csv` | 対象日のカレンダーを使う。年や閏年によって曜日・祝日の配置が変わる | [09](09_forecast_coverage.ipynb), [14](14_horizon_leakage.ipynb) |
| 店舗属性・立地 | `stores.csv` の店舗、立地、床面積、開店日 | 店舗別・立地別の履歴水準を比較。店舗固有の変化は一律の伸び率では捉えにくい | [01](01_sales.ipynb), [05](05_demand_drivers.ipynb), [13](13_revenue_decomposition.ipynb) |
| 座席・卓構成 | `seat_layout.csv` の対象日に有効なレイアウト | 対象期間の予定を使用。店舗×日で有効期間を両端含めて結合し、重複・欠落を確認。過去の待ち・占有実績を未来同日に使わない | [02](02_context.ipynb), [08](08_change_events.ipynb), [17](17_capacity_timeline.ipynb) |
| 商品属性 | `menus.csv` のカテゴリ、種類、販売時間帯、季節・コラボ情報 | 商品ごとに欠損の意味が異なる。季節・企画限定品を定番品と同列に扱わない | [01](01_sales.ipynb), [02](02_context.ipynb), [18](18_segments_dayparts.ipynb) |
| 販売可能性 | `menu_catalog_history.csv` の期間と `campaign_calendar.csv` の企画参加店舗 | コラボは期間と対象店舗の両方を照合。過去に条件外の正売上はなかったが、将来の絶対的保証とは区別する | [05](05_demand_drivers.ipynb), [09](09_forecast_coverage.ipynb), [19](19_zero_sales.ipynb) |
| 価格計画 | `price_history.csv` の対象店舗・日・商品の割引前販売価格 | `all`・`\|`区切りを展開し、期間の両端を含める。価格と販売額の過去の差は価格の因果効果とは限らない | [02](02_context.ipynb), [05](05_demand_drivers.ipynb), [08](08_change_events.ipynb), [15](15_order_integrity.ipynb) |
| 施策予定 | `campaign_calendar.csv` の日・対象店舗・商品群・割引率 | 同日に複数施策が重なり得る。予定の有無とクーポンの実際の利用を分け、単純結合による行増加を防ぐ | [05](05_demand_drivers.ipynb), [10](10_context_comparisons.ipynb), [15](15_order_integrity.ipynb), [21](21_coupon_exposure.ipynb) |
| 天候平年値 | `calendar.csv` の平年気温・雨確率・平年降水量 | 2025年にも配布。`weather_observed`・観測気温・観測降水量は2025年に未観測で、同日の予測に使えない | [02](02_context.ipynb), [09](09_forecast_coverage.ipynb), [10](10_context_comparisons.ipynb) |
| 過去売上・曜日別傾向 | `train.csv` から基準日以前の店舗・商品・曜日別統計を作る | 統計計算の基準日を固定。2025年の7日前実績は最初の7日しか利用できず、364日前も年末1日は未来。予測値を再帰利用する場合は実績と区別する | [07](07_validation.ipynb), [14](14_horizon_leakage.ipynb) |
| 未観測の店舗・商品 | `train.csv` と販売・企画予定を比較 | 2025年の販売可能枠に、過去の同一店舗・商品・曜日の正売上がない324行がある。商品・他店舗・カテゴリ等の過去統計による補完は時間順に検証する | [09](09_forecast_coverage.ipynb), [20](20_cold_start.ipynb) |
| 過去の客層・来店・待ち | `logs.zip` の来店明細を基準日以前で集約 | 来店・着席・待ち・離脱は同時点の実績で、将来同日の値は未観測。卓の定員と着席人数は異なる | [06](06_logs_deep_dive.ipynb), [13](13_revenue_decomposition.ipynb), [16](16_visit_integrity.ipynb), [17](17_capacity_timeline.ipynb) |
| 過去の数量・クーポン利用 | `logs.zip` の注文明細を基準日以前で集約 | 注文数量・実際の値引きは未来同日に利用不可。予定された価格やキャンペーンと区別する | [06](06_logs_deep_dive.ipynb), [08](08_change_events.ipynb), [15](15_order_integrity.ipynb) |
| 商品間の関係 | 会計単位の商品共起を過去データで集約 | 全期間の lift は共通する販売時期や店舗の影響を受ける。対象日の他商品の実績売上は利用不可 | [11](11_baskets.ipynb), [12](12_baskets_adjusted.ipynb) |

## 検証の基本

- 2024年の通年検証では、2023年末を予測基準日にする。2024年の既知計画を使う場合は、その計画が2023年末に利用可能だったという仮定を明示する。
- 提出ガイドに従い、2022年の店舗・商品別日平均から2023年を予測した誤差で、2024年検証用の6層のスケールを作る。配布済み重みには2024年の正解が含まれるため、2024年検証にそのまま使わない。[07_validation.ipynb](07_validation.ipynb) に計算例がある。
- 2024年だけで多数の候補を選ぶと検証期間に過学習する。層別・店舗別・月別の誤差や、検証の仮定を併記して判断する。
- 天候・施策・価格・座席などの前後差や共起は交絡を含む。EDAで見えた差を因果効果として扱わず、特徴量としての有効性を時間順に検証する。
