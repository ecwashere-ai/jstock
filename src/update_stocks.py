#!/usr/bin/env python3
"""日本株テクニカル指標スクリーニング用データ更新スクリプト。

yfinance で過去50営業日の株価を取得し、RSI(14) や 25日移動平均乖離率
などを算出して data/stocks.json に保存する。GitHub Actions から
毎営業日 11:30 JST に実行される想定。
"""

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 対象銘柄リスト
# 銘柄を追加したい場合はこの配列に {"code": "XXXX.T", "name": "銘柄名", "market": "市場"}
# の形式で1行追加するだけでよい。
# ---------------------------------------------------------------------------
STOCKS = [
    {"code": "7203.T", "name": "トヨタ自動車",           "market": "東証プライム"},
    {"code": "6758.T", "name": "ソニーグループ",         "market": "東証プライム"},
    {"code": "8306.T", "name": "三菱UFJフィナンシャルG", "market": "東証プライム"},
    {"code": "7974.T", "name": "任天堂",                 "market": "東証プライム"},
    {"code": "6861.T", "name": "キーエンス",             "market": "東証プライム"},
    {"code": "9984.T", "name": "ソフトバンクグループ",   "market": "東証プライム"},
    {"code": "9983.T", "name": "ファーストリテイリング", "market": "東証プライム"},
    {"code": "1605.T", "name": "INPEX",                  "market": "東証プライム"},
    {"code": "8058.T", "name": "三菱商事",               "market": "東証プライム"},
    {"code": "8267.T", "name": "イオン",                 "market": "東証プライム"},
    {"code": "8035.T", "name": "東京エレクトロン",       "market": "東証プライム"},
    {"code": "6501.T", "name": "日立製作所",             "market": "東証プライム"},
    {"code": "4063.T", "name": "信越化学工業",           "market": "東証プライム"},
    {"code": "9432.T", "name": "日本電信電話",           "market": "東証プライム"},
    {"code": "8316.T", "name": "三井住友フィナンシャルG", "market": "東証プライム"},
    {"code": "4502.T", "name": "武田薬品工業",           "market": "東証プライム"},
    {"code": "6902.T", "name": "デンソー",               "market": "東証プライム"},
    {"code": "6098.T", "name": "リクルートホールディングス", "market": "東証プライム"},
]

RSI_PERIOD = 14
MA_PERIOD = 25
HISTORY_DAYS = 50

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stocks.json"


def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float:
    """Wilder 平滑化による RSI を返す。データ不足時は NaN。"""
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - 100 / (1 + rs))


def build_signal(rsi: float) -> str:
    """RSI 値に基づく客観的なテクニカル状態のテキストを返す。"""
    if math.isnan(rsi):
        return "データ不足"
    if rsi <= 30:
        return "売られすぎ水準（RSI30以下）"
    if rsi >= 70:
        return "買われすぎ水準（RSI70以上）"
    return "シグナルなし"


def round_or_none(value: float, digits: int = 2):
    """NaN を None に変換しつつ丸める（JSON に null として出力するため）。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def fetch_stock(entry: dict) -> dict | None:
    code = entry["code"]
    try:
        history = yf.Ticker(code).history(
            period=f"{HISTORY_DAYS * 2}d",  # 休場日を考慮して多めに取得
            interval="1d",
            auto_adjust=True,
        )
    except Exception as exc:  # ネットワーク・API 障害時は当該銘柄をスキップ
        print(f"[WARN] {code}: fetch failed: {exc}")
        return None

    history = history.dropna(subset=["Close"]).tail(HISTORY_DAYS)
    if len(history) < 2:
        print(f"[WARN] {code}: not enough data ({len(history)} rows)")
        return None

    close = history["Close"]
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = (price - prev) / prev * 100 if prev else float("nan")

    rsi = calc_rsi(close)

    if len(close) >= MA_PERIOD:
        ma25 = float(close.rolling(MA_PERIOD).mean().iloc[-1])
        ma_dev_pct = (price - ma25) / ma25 * 100 if ma25 else float("nan")
    else:
        ma_dev_pct = float("nan")

    return {
        "code": code.replace(".T", ""),
        "ticker": code,
        "name": entry["name"],
        "market": entry["market"],
        "price": round_or_none(price),
        "change_pct": round_or_none(change_pct),
        "rsi": round_or_none(rsi, 1),
        "ma25_dev_pct": round_or_none(ma_dev_pct),
        "signal": build_signal(rsi if rsi is not None else float("nan")),
    }


def main() -> None:
    results = []
    for entry in STOCKS:
        row = fetch_stock(entry)
        if row is not None:
            results.append(row)
            print(f"[OK] {row['ticker']}: price={row['price']} rsi={row['rsi']}")

    jst = timezone(timedelta(hours=9))
    payload = {
        "updated_at": datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "count": len(results),
        "stocks": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(results)} stocks -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
