# JP Stock Screener

日本株のテクニカル指標を毎営業日自動取得・計算し、静的ダッシュボードで閲覧できる個人用スクリーニングシステム。外部の有料APIは使わず、GitHub Actions の Cron と GitHub Pages のみで完結します。

## 構成

| ファイル | 役割 |
|---|---|
| `.github/workflows/update_data.yml` | 平日 11:30 JST（前場引け）に自動実行される GitHub Actions ワークフロー |
| `src/update_stocks.py` | yfinance で株価を取得し、RSI(14)・25日移動平均乖離率などを計算して `data/stocks.json` に保存 |
| `src/index.html` | Tailwind CSS (CDN) + バニラJS のビルド不要ダッシュボード（ダークモード） |
| `data/stocks.json` | 計算済みデータ（Actions が自動更新） |

## セットアップ

1. リポジトリの **Settings → Pages** で、Source を「Deploy from a branch」、ブランチをメインブランチ（フォルダは `/ (root)`）に設定
2. **Actions** タブから `Update Stock Data` ワークフローを手動実行（workflow_dispatch）して初回データを生成
3. `https://<ユーザー名>.github.io/<リポジトリ名>/src/` にアクセス

以降は平日 11:30 JST に自動でデータが更新されます。

## 銘柄の追加

`src/update_stocks.py` の `STOCKS` 配列に1行追加するだけです。

```python
{"code": "XXXX.T", "name": "銘柄名", "market": "市場名"},
```

## 表示指標

- 現在値（終値）
- 前日比（騰落率 %）
- RSI (14日, Wilder 平滑化)
- 25日移動平均乖離率 (%)
- RSI 水準に基づく客観的なテクニカル状態表示

## 免責

本システムはテクニカル指標の数値を客観的に表示するものであり、投資勧誘や売買推奨を行うものではありません。データは Yahoo Finance 由来で遅延・欠損を含む場合があります。投資判断はご自身の責任で行ってください。個人利用限定。
