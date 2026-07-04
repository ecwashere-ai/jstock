#!/usr/bin/env python3
"""日本株テクニカル指標スクリーニング用データ更新スクリプト。

yfinance で過去約2年分の株価を取得し、RSI・MACD・移動平均トレンド・
ボリンジャーバンド・出来高など複数のテクニカル指標を算出する。
さらに、それぞれの銘柄自身の過去データにおいて「現在と似たテクニカル状態」
から一定日数後に株価が上昇していた割合（統計的上昇確率）を集計する。
結果は data/stocks.json に保存する。GitHub Actions から毎営業日
11:30 JST に実行される想定。

※ 統計的上昇確率はあくまで過去の値動きの頻度を機械的に集計した参考値であり、
   将来の値動きを保証・予測するものではない。売買の推奨を行うものではない。
"""

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
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
    {"code": "9433.T", "name": "KDDI",                   "market": "東証プライム"},
    {"code": "8001.T", "name": "伊藤忠商事",             "market": "東証プライム"},
    {"code": "8031.T", "name": "三井物産",               "market": "東証プライム"},
    {"code": "8766.T", "name": "東京海上ホールディングス", "market": "東証プライム"},
    {"code": "8411.T", "name": "みずほフィナンシャルG",  "market": "東証プライム"},
    {"code": "6954.T", "name": "ファナック",             "market": "東証プライム"},
    {"code": "6367.T", "name": "ダイキン工業",           "market": "東証プライム"},
    {"code": "6981.T", "name": "村田製作所",             "market": "東証プライム"},
    {"code": "6857.T", "name": "アドバンテスト",         "market": "東証プライム"},
    {"code": "6146.T", "name": "ディスコ",               "market": "東証プライム"},
    {"code": "6503.T", "name": "三菱電機",               "market": "東証プライム"},
    {"code": "6702.T", "name": "富士通",                 "market": "東証プライム"},
    {"code": "7011.T", "name": "三菱重工業",             "market": "東証プライム"},
    {"code": "7267.T", "name": "ホンダ",                 "market": "東証プライム"},
    {"code": "7741.T", "name": "HOYA",                   "market": "東証プライム"},
    {"code": "4568.T", "name": "第一三共",               "market": "東証プライム"},
    {"code": "4519.T", "name": "中外製薬",               "market": "東証プライム"},
    {"code": "4503.T", "name": "アステラス製薬",         "market": "東証プライム"},
    {"code": "2914.T", "name": "日本たばこ産業",         "market": "東証プライム"},
    {"code": "3382.T", "name": "セブン&アイ・ホールディングス", "market": "東証プライム"},
    {"code": "4661.T", "name": "オリエンタルランド",     "market": "東証プライム"},
    {"code": "5401.T", "name": "日本製鉄",               "market": "東証プライム"},
    {"code": "9101.T", "name": "日本郵船",               "market": "東証プライム"},
    {"code": "8591.T", "name": "オリックス",             "market": "東証プライム"},
    {"code": "7751.T", "name": "キヤノン",               "market": "東証プライム"},
    {"code": "6301.T", "name": "小松製作所",             "market": "東証プライム"},
]

# ---------------------------------------------------------------------------
# 指標パラメータ
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
MA_SHORT = 25
MA_LONG = 75
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_NUM_STD = 2
VOLUME_MA_PERIOD = 20
VOLUME_SURGE_RATIO = 1.5

FETCH_PERIOD = "2y"        # バックテスト用に約2年分を取得
FORWARD_DAYS = 5           # 何営業日後の値動きを見るか
MIN_BACKTEST_SAMPLES = 5   # この件数未満の場合は確率を「参考不可」扱いにする

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stocks.json"

# 総合スコア（-5〜+5）を5段階に分類する際のラベル（客観的なテクニカル用語のみ使用）
SCORE_LABELS = {
    "strong_bull": "強気シグナル優勢",
    "mild_bull":   "やや強気",
    "neutral":     "中立",
    "mild_bear":   "やや弱気",
    "strong_bear": "弱気シグナル優勢",
}


def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder 平滑化による RSI の時系列を返す。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def calc_macd(close: pd.Series):
    """MACD 線・シグナル線・ヒストグラムの時系列を返す。"""
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger_percent_b(close: pd.Series, period: int = BB_PERIOD, num_std: float = BB_NUM_STD) -> pd.Series:
    """ボリンジャーバンド %B（0=下限バンド, 1=上限バンド）の時系列を返す。"""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = upper - lower
    percent_b = (close - lower) / band_width
    return percent_b.where(band_width != 0)


def build_signal(rsi: float) -> str:
    """RSI 値に基づく客観的なテクニカル状態のテキストを返す。"""
    if rsi is None or math.isnan(rsi):
        return "データ不足"
    if rsi <= 30:
        return "売られすぎ水準（RSI30以下）"
    if rsi >= 70:
        return "買われすぎ水準（RSI70以上）"
    return "シグナルなし"


def round_or_none(value, digits: int = 2):
    """NaN を None に変換しつつ丸める（JSON に null として出力するため）。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def bucket_from_score(score: float):
    """総合スコア（-5〜+5）を5段階のバケットキーに分類する。NaN は None。"""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    if score >= 3:
        return "strong_bull"
    if score >= 1:
        return "mild_bull"
    if score <= -3:
        return "strong_bear"
    if score <= -1:
        return "mild_bear"
    return "neutral"


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """全期間の終値・出来高から各指標とテクニカル投票列を計算したDataFrameを返す。"""
    close = history["Close"]
    volume = history["Volume"]

    df = pd.DataFrame(index=history.index)
    df["close"] = close
    df["rsi"] = calc_rsi(close)
    _, _, macd_hist = calc_macd(close)
    df["macd_hist"] = macd_hist
    df["ma_short"] = close.rolling(MA_SHORT).mean()
    df["ma_long"] = close.rolling(MA_LONG).mean()
    df["bb_percent_b"] = calc_bollinger_percent_b(close)
    vol_ma = volume.rolling(VOLUME_MA_PERIOD).mean()
    df["volume_ratio"] = volume / vol_ma
    daily_change = close.diff()

    # --- 各指標の投票（+1: 強気寄り, -1: 弱気寄り, 0: 中立）。NaN は判定不能として NaN のまま伝播させる ---
    vote_rsi = pd.Series(
        np.select([df["rsi"] <= 30, df["rsi"] >= 70], [1, -1], default=0),
        index=df.index, dtype=float,
    ).where(df["rsi"].notna())

    vote_macd = pd.Series(
        np.select([df["macd_hist"] > 0, df["macd_hist"] < 0], [1, -1], default=0),
        index=df.index, dtype=float,
    ).where(df["macd_hist"].notna())

    ma_trend_up = (close > df["ma_short"]) & (df["ma_short"] > df["ma_long"])
    ma_trend_down = (close < df["ma_short"]) & (df["ma_short"] < df["ma_long"])
    vote_ma = pd.Series(
        np.select([ma_trend_up, ma_trend_down], [1, -1], default=0),
        index=df.index, dtype=float,
    ).where(df["ma_long"].notna())

    vote_bb = pd.Series(
        np.select([df["bb_percent_b"] < 0.2, df["bb_percent_b"] > 0.8], [1, -1], default=0),
        index=df.index, dtype=float,
    ).where(df["bb_percent_b"].notna())

    volume_surge = df["volume_ratio"] > VOLUME_SURGE_RATIO
    vote_volume = pd.Series(
        np.select(
            [volume_surge & (daily_change > 0), volume_surge & (daily_change < 0)],
            [1, -1], default=0,
        ),
        index=df.index, dtype=float,
    ).where(df["volume_ratio"].notna() & daily_change.notna())

    df["composite_score"] = vote_rsi + vote_macd + vote_ma + vote_bb + vote_volume
    df["forward_up"] = close.shift(-FORWARD_DAYS) > close
    return df


def backtest_up_probability(df: pd.DataFrame):
    """今日の総合スコアと同じバケットが過去何回発生し、その後 FORWARD_DAYS 営業日後に
    上昇していた割合（%）を返す。サンプル不足の場合は (None, sample_size) を返す。
    """
    today_score = df["composite_score"].iloc[-1]
    today_bucket = bucket_from_score(today_score)
    if today_bucket is None:
        return None, 0, today_score, None

    hist = df.iloc[:-1].dropna(subset=["composite_score", "forward_up"])
    hist_bucket = hist["composite_score"].apply(bucket_from_score)
    matched = hist.loc[hist_bucket == today_bucket, "forward_up"]

    sample_size = int(matched.shape[0])
    if sample_size < MIN_BACKTEST_SAMPLES:
        return None, sample_size, today_score, today_bucket

    up_probability = float(matched.mean() * 100)
    return up_probability, sample_size, today_score, today_bucket


def fetch_stock(entry: dict) -> dict | None:
    code = entry["code"]
    try:
        history = yf.Ticker(code).history(
            period=FETCH_PERIOD,
            interval="1d",
            auto_adjust=True,
        )
    except Exception as exc:  # ネットワーク・API 障害時は当該銘柄をスキップ
        print(f"[WARN] {code}: fetch failed: {exc}")
        return None

    history = history.dropna(subset=["Close"])
    if len(history) < 2:
        print(f"[WARN] {code}: not enough data ({len(history)} rows)")
        return None

    close = history["Close"]
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = (price - prev) / prev * 100 if prev else float("nan")

    df = compute_indicators(history)
    rsi = df["rsi"].iloc[-1]

    if len(close) >= MA_SHORT:
        ma25 = float(close.rolling(MA_SHORT).mean().iloc[-1])
        ma_dev_pct = (price - ma25) / ma25 * 100 if ma25 else float("nan")
    else:
        ma_dev_pct = float("nan")

    up_probability, sample_size, composite_score, bucket = backtest_up_probability(df)
    composite_score_int = None if bucket is None else int(composite_score)
    composite_label = SCORE_LABELS[bucket] if bucket is not None else "データ不足"

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
        "macd_hist": round_or_none(df["macd_hist"].iloc[-1], 2),
        "bb_percent_b": round_or_none(df["bb_percent_b"].iloc[-1] * 100, 1) if pd.notna(df["bb_percent_b"].iloc[-1]) else None,
        "volume_ratio": round_or_none(df["volume_ratio"].iloc[-1], 2),
        "composite_score": composite_score_int,
        "composite_label": composite_label,
        "up_probability": round_or_none(up_probability, 1),
        "up_probability_samples": sample_size,
        "forward_days": FORWARD_DAYS,
    }


def main() -> None:
    results = []
    for entry in STOCKS:
        row = fetch_stock(entry)
        if row is not None:
            results.append(row)
            print(
                f"[OK] {row['ticker']}: price={row['price']} rsi={row['rsi']} "
                f"score={row['composite_score']} up_prob={row['up_probability']}"
                f"(n={row['up_probability_samples']})"
            )

    jst = timezone(timedelta(hours=9))
    payload = {
        "updated_at": datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "count": len(results),
        "forward_days": FORWARD_DAYS,
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
