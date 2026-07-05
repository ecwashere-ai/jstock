#!/usr/bin/env python3
"""日本株テクニカル指標スクリーニング用データ更新スクリプト。

yfinance で過去約2年分の株価を取得し、RSI・MACD・移動平均トレンド・
ボリンジャーバンド・出来高・ストキャスティクスなど複数のテクニカル指標を
内部的に算出し、それらを合成した単一の総合スコアに集約する。
さらに、それぞれの銘柄自身の過去データにおいて「現在と似たテクニカル状態」
から一定日数後に株価が上昇していた割合（統計的上昇確率）を集計する。
結果は data/stocks.json に保存する。GitHub Actions から毎営業日
11:30 JST に実行される想定。

対象銘柄は data/universe_prime.json（fetch_universe.py が JPX公式の
上場銘柄一覧から生成する東証プライム全銘柄リスト）を優先的に使用し、
これが存在しない場合のみ下記の FALLBACK_STOCKS（動作確認用の主要銘柄）
にフォールバックする。

銘柄数が多い場合の Yahoo Finance 側のレート制限を避けるため、
株価取得は CHUNK_SIZE 件ずつまとめて yf.download で並行取得する。
なお「上位銘柄に絞る」のは最終的な出力（data/stocks.json）のみであり、
そもそも上位を発見するために毎回プライム市場全銘柄をスキャンする必要が
あるため、内部の取得・計算処理自体は全銘柄が対象のまま。

最終的に data/stocks.json へ保存するのは、総合スコアが最も強い
「買い」候補 TOP_N 銘柄と「売り」候補 TOP_N 銘柄（合計最大 TOP_N*2 銘柄、
既定40銘柄）のみに絞り込む。スマートフォンでの一覧性と読み込み速度を
優先するための設計。

個別銘柄のテクニカル指標に加えて、以下も総合スコアに反映する。
- 市場全体: 日経平均(^N225)自身のRSI・MACD・移動平均トレンド・
  ボリンジャーバンド・ストキャスティクスから市場全体の地合いを判定し、
  全銘柄に共通のボーナス/ペナルティとして加算する。
- セクター動向: 本日スキャンした同一セクター銘柄の平均スコアを算出し、
  セクター全体が堅調/軟調であれば当該銘柄のスコアに反映する。

また、同一銘柄の過去統計から算出した「目標株価」（参考値）と、
直近の値動きを示す簡易チャート用の株価系列（spark）も出力する。

※ 統計的上昇確率・総合スコア・目標株価はあくまで過去の値動きの頻度や
   平均変動幅を機械的に集計した参考値であり、将来の値動きを保証・予測
   するものではない。個人の情報整理を目的とした参考情報であり、
   投資助言業の登録を受けた者による助言ではない。
"""

import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe_prime.json"

# ---------------------------------------------------------------------------
# フォールバック銘柄リスト（data/universe_prime.json が無い/読み込めない場合に使用）
# 銘柄を追加したい場合はこの配列に
# {"code": "XXXX.T", "name": "銘柄名", "market": "市場", "sector": "33業種区分"}
# の形式で1行追加するだけでよい。
# ---------------------------------------------------------------------------
FALLBACK_STOCKS = [
    {"code": "7203.T", "name": "トヨタ自動車",           "market": "東証プライム", "sector": "輸送用機器"},
    {"code": "6758.T", "name": "ソニーグループ",         "market": "東証プライム", "sector": "電気機器"},
    {"code": "8306.T", "name": "三菱UFJフィナンシャルG", "market": "東証プライム", "sector": "銀行業"},
    {"code": "7974.T", "name": "任天堂",                 "market": "東証プライム", "sector": "その他製品"},
    {"code": "6861.T", "name": "キーエンス",             "market": "東証プライム", "sector": "電気機器"},
    {"code": "9984.T", "name": "ソフトバンクグループ",   "market": "東証プライム", "sector": "情報・通信業"},
    {"code": "9983.T", "name": "ファーストリテイリング", "market": "東証プライム", "sector": "小売業"},
    {"code": "1605.T", "name": "INPEX",                  "market": "東証プライム", "sector": "鉱業"},
    {"code": "8058.T", "name": "三菱商事",               "market": "東証プライム", "sector": "卸売業"},
    {"code": "8267.T", "name": "イオン",                 "market": "東証プライム", "sector": "小売業"},
    {"code": "8035.T", "name": "東京エレクトロン",       "market": "東証プライム", "sector": "電気機器"},
    {"code": "6501.T", "name": "日立製作所",             "market": "東証プライム", "sector": "電気機器"},
    {"code": "4063.T", "name": "信越化学工業",           "market": "東証プライム", "sector": "化学"},
    {"code": "9432.T", "name": "日本電信電話",           "market": "東証プライム", "sector": "情報・通信業"},
    {"code": "8316.T", "name": "三井住友フィナンシャルG", "market": "東証プライム", "sector": "銀行業"},
    {"code": "4502.T", "name": "武田薬品工業",           "market": "東証プライム", "sector": "医薬品"},
    {"code": "6902.T", "name": "デンソー",               "market": "東証プライム", "sector": "輸送用機器"},
    {"code": "6098.T", "name": "リクルートホールディングス", "market": "東証プライム", "sector": "サービス業"},
    {"code": "9433.T", "name": "KDDI",                   "market": "東証プライム", "sector": "情報・通信業"},
    {"code": "8001.T", "name": "伊藤忠商事",             "market": "東証プライム", "sector": "卸売業"},
    {"code": "8031.T", "name": "三井物産",               "market": "東証プライム", "sector": "卸売業"},
    {"code": "8766.T", "name": "東京海上ホールディングス", "market": "東証プライム", "sector": "保険業"},
    {"code": "8411.T", "name": "みずほフィナンシャルG",  "market": "東証プライム", "sector": "銀行業"},
    {"code": "6954.T", "name": "ファナック",             "market": "東証プライム", "sector": "電気機器"},
    {"code": "6367.T", "name": "ダイキン工業",           "market": "東証プライム", "sector": "機械"},
    {"code": "6981.T", "name": "村田製作所",             "market": "東証プライム", "sector": "電気機器"},
    {"code": "6857.T", "name": "アドバンテスト",         "market": "東証プライム", "sector": "電気機器"},
    {"code": "6146.T", "name": "ディスコ",               "market": "東証プライム", "sector": "機械"},
    {"code": "6503.T", "name": "三菱電機",               "market": "東証プライム", "sector": "電気機器"},
    {"code": "6702.T", "name": "富士通",                 "market": "東証プライム", "sector": "電気機器"},
    {"code": "7011.T", "name": "三菱重工業",             "market": "東証プライム", "sector": "機械"},
    {"code": "7267.T", "name": "ホンダ",                 "market": "東証プライム", "sector": "輸送用機器"},
    {"code": "7741.T", "name": "HOYA",                   "market": "東証プライム", "sector": "精密機器"},
    {"code": "4568.T", "name": "第一三共",               "market": "東証プライム", "sector": "医薬品"},
    {"code": "4519.T", "name": "中外製薬",               "market": "東証プライム", "sector": "医薬品"},
    {"code": "4503.T", "name": "アステラス製薬",         "market": "東証プライム", "sector": "医薬品"},
    {"code": "2914.T", "name": "日本たばこ産業",         "market": "東証プライム", "sector": "食料品"},
    {"code": "3382.T", "name": "セブン&アイ・ホールディングス", "market": "東証プライム", "sector": "小売業"},
    {"code": "4661.T", "name": "オリエンタルランド",     "market": "東証プライム", "sector": "サービス業"},
    {"code": "5401.T", "name": "日本製鉄",               "market": "東証プライム", "sector": "鉄鋼"},
    {"code": "9101.T", "name": "日本郵船",               "market": "東証プライム", "sector": "海運業"},
    {"code": "8591.T", "name": "オリックス",             "market": "東証プライム", "sector": "その他金融業"},
    {"code": "7751.T", "name": "キヤノン",               "market": "東証プライム", "sector": "電気機器"},
    {"code": "6301.T", "name": "小松製作所",             "market": "東証プライム", "sector": "機械"},
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
STOCH_PERIOD = 14
STOCH_SMOOTH = 3

FETCH_PERIOD = "2y"        # バックテスト用に約2年分を取得
FORWARD_DAYS = 5           # 何営業日後の値動きを見るか
MIN_BACKTEST_SAMPLES = 5   # この件数未満の場合は確率を「参考不可」扱いにする

# 東証プライム全銘柄など件数が多い場合、Yahoo Finance への同時リクエストが
# 集中しすぎるとレート制限にかかりやすいため、CHUNK_SIZE件ずつに分割して
# 取得し、チャンク間に短い待機を挟む。
CHUNK_SIZE = 150
CHUNK_DELAY_SEC = 2

# 最終出力に残す「買い」候補・「売り」候補それぞれの件数（合計最大 TOP_N*2 銘柄）
TOP_N = 20

# 市場全体の地合い判定に使う指数（日経平均株価）
MARKET_INDEX_TICKER = "^N225"

# セクター平均スコアがこの値以上/以下の場合にセクター全体を強気/弱気とみなす
SECTOR_VOTE_THRESHOLD = 1.0

# カード内の簡易チャート（スパークライン）に使う直近営業日数
SPARKLINE_DAYS = 30

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stocks.json"

# 総合スコア（-6〜+6）を5段階に分類する際のラベル（客観的なテクニカル用語のみ使用）
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


def calc_stochastic_k(history: pd.DataFrame, period: int = STOCH_PERIOD, smooth: int = STOCH_SMOOTH) -> pd.Series:
    """ストキャスティクス %K（Slow, 0〜100）の時系列を返す。"""
    low_min = history["Low"].rolling(period).min()
    high_max = history["High"].rolling(period).max()
    band_width = high_max - low_min
    raw_k = (history["Close"] - low_min) / band_width * 100
    k = raw_k.rolling(smooth).mean()
    return k.where(band_width != 0)


def build_signal(rsi: float) -> str:
    """RSI 値に基づく客観的なテクニカル状態のテキストを返す。"""
    if rsi is None or math.isnan(rsi):
        return "データ不足"
    if rsi <= 30:
        return "売られすぎ水準（RSI30以下）"
    if rsi >= 70:
        return "買われすぎ水準（RSI70以上）"
    return "シグナルなし"


def build_reasons(latest: pd.Series) -> list[str]:
    """直近1営業日の指標値から、総合スコアの根拠となった客観的なテクニカル
    状態を短い日本語の文でリストアップする（断定的な売買表現は使わない）。
    """
    reasons = []

    rsi = latest["rsi"]
    if pd.notna(rsi):
        if rsi <= 30:
            reasons.append(f"RSIが{rsi:.1f}と売られすぎ水準")
        elif rsi >= 70:
            reasons.append(f"RSIが{rsi:.1f}と買われすぎ水準")

    macd_hist = latest["macd_hist"]
    if pd.notna(macd_hist):
        if macd_hist > 0:
            reasons.append("MACDがシグナル線を上回り上昇モメンタム")
        elif macd_hist < 0:
            reasons.append("MACDがシグナル線を下回り下降モメンタム")

    close, ma_short, ma_long = latest["close"], latest["ma_short"], latest["ma_long"]
    if pd.notna(ma_short) and pd.notna(ma_long):
        if close > ma_short > ma_long:
            reasons.append("25日線が75日線を上回る上昇トレンド")
        elif close < ma_short < ma_long:
            reasons.append("25日線が75日線を下回る下降トレンド")

    bb = latest["bb_percent_b"]
    if pd.notna(bb):
        if bb < 0.2:
            reasons.append("ボリンジャーバンド下限に接近")
        elif bb > 0.8:
            reasons.append("ボリンジャーバンド上限に接近")

    vol_ratio = latest["volume_ratio"]
    if pd.notna(vol_ratio) and vol_ratio > VOLUME_SURGE_RATIO:
        reasons.append(f"出来高が平常の{vol_ratio:.1f}倍に急増")

    stoch = latest["stoch_k"]
    if pd.notna(stoch):
        if stoch < 20:
            reasons.append("ストキャスティクスが売られすぎ水準")
        elif stoch > 80:
            reasons.append("ストキャスティクスが買われすぎ水準")

    return reasons


def round_or_none(value, digits: int = 2):
    """NaN を None に変換しつつ丸める（JSON に null として出力するため）。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def bucket_from_score(score: float):
    """総合スコア（-6〜+6）を5段階のバケットキーに分類する。NaN は None。"""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    if score >= 4:
        return "strong_bull"
    if score >= 2:
        return "mild_bull"
    if score <= -4:
        return "strong_bear"
    if score <= -2:
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
    df["stoch_k"] = calc_stochastic_k(history)
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

    vote_stoch = pd.Series(
        np.select([df["stoch_k"] < 20, df["stoch_k"] > 80], [1, -1], default=0),
        index=df.index, dtype=float,
    ).where(df["stoch_k"].notna())

    df["composite_score"] = vote_rsi + vote_macd + vote_ma + vote_bb + vote_volume + vote_stoch
    df["forward_up"] = close.shift(-FORWARD_DAYS) > close
    df["forward_return_pct"] = (close.shift(-FORWARD_DAYS) - close) / close * 100
    return df


def backtest_up_probability(df: pd.DataFrame):
    """今日の総合スコアと同じバケットが過去何回発生し、その後 FORWARD_DAYS 営業日後に
    上昇していた割合（%）と平均変動率（目標株価の算出に使う）を返す。
    サンプル不足の場合は up_probability / expected_return_pct に None を返す。
    """
    today_score = df["composite_score"].iloc[-1]
    today_bucket = bucket_from_score(today_score)
    if today_bucket is None:
        return None, None, 0, today_score, None

    hist = df.iloc[:-1].dropna(subset=["composite_score", "forward_up", "forward_return_pct"])
    hist_bucket = hist["composite_score"].apply(bucket_from_score)
    matched = hist.loc[hist_bucket == today_bucket]

    sample_size = int(matched.shape[0])
    if sample_size < MIN_BACKTEST_SAMPLES:
        return None, None, sample_size, today_score, today_bucket

    up_probability = float(matched["forward_up"].mean() * 100)
    expected_return_pct = float(matched["forward_return_pct"].mean())
    return up_probability, expected_return_pct, sample_size, today_score, today_bucket


def select_top_candidates(rows: list[dict], top_n: int = TOP_N) -> list[dict]:
    """総合スコアが強い「買い」候補・「売り」候補をそれぞれ上位 top_n 件選ぶ。
    同点の場合は統計的上昇確率が50%からより離れている（自信度が高い）方を優先する。
    """
    def confidence(row: dict) -> float:
        prob = row["up_probability"]
        return abs(prob - 50) if prob is not None else -1.0

    valid = [r for r in rows if r["composite_score"] is not None]
    bullish = [r for r in valid if r["composite_score"] > 0]
    bearish = [r for r in valid if r["composite_score"] < 0]

    bullish.sort(key=lambda r: (r["composite_score"], confidence(r)), reverse=True)
    bearish.sort(key=lambda r: (-r["composite_score"], confidence(r)), reverse=True)

    for row in bullish:
        row["call"] = "買い"
    for row in bearish:
        row["call"] = "売り"

    return bullish[:top_n] + bearish[:top_n]


def compute_market_context(history: pd.DataFrame):
    """市場全体（日経平均株価）の地合いを判定する。

    個別銘柄と異なり指数は出来高データが信頼できない（0やNaNが多い）ため、
    出来高急増の指標は使わず、RSI・MACD・移動平均トレンド・ボリンジャー
    バンド・ストキャスティクスの5指標のみで -1(弱気)/0(中立)/+1(強気) の
    市場ボーナスと、その理由テキストを返す。
    """
    close = history["Close"]
    rsi = calc_rsi(close).iloc[-1]
    _, _, macd_hist_series = calc_macd(close)
    macd_last = macd_hist_series.iloc[-1]
    ma_short = close.rolling(MA_SHORT).mean().iloc[-1]
    ma_long = close.rolling(MA_LONG).mean().iloc[-1]
    last_close = close.iloc[-1]
    bb = calc_bollinger_percent_b(close).iloc[-1]
    stoch = calc_stochastic_k(history).iloc[-1] if {"High", "Low"}.issubset(history.columns) else float("nan")

    votes = []
    if pd.notna(rsi):
        votes.append(1 if rsi <= 30 else -1 if rsi >= 70 else 0)
    if pd.notna(macd_last):
        votes.append(1 if macd_last > 0 else -1 if macd_last < 0 else 0)
    if pd.notna(ma_short) and pd.notna(ma_long):
        if last_close > ma_short > ma_long:
            votes.append(1)
        elif last_close < ma_short < ma_long:
            votes.append(-1)
        else:
            votes.append(0)
    if pd.notna(bb):
        votes.append(1 if bb < 0.2 else -1 if bb > 0.8 else 0)
    if pd.notna(stoch):
        votes.append(1 if stoch < 20 else -1 if stoch > 80 else 0)

    if not votes:
        return 0, "日経平均のデータ不足のため市場全体の判定なし"

    total = sum(votes)
    if total >= 2:
        return 1, "日経平均は上昇トレンドで市場全体は追い風"
    if total <= -2:
        return -1, "日経平均は下落トレンドで市場全体は逆風"
    return 0, "日経平均は方向感に乏しく市場全体は中立"


def compute_sector_averages(rows: list[dict]) -> dict[str, float]:
    """本日スキャンした銘柄を業種ごとにグルーピングし、平均の技術スコアを返す。"""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["composite_score"] is not None:
            groups[row["sector"]].append(row["composite_score"])
    return {sector: sum(scores) / len(scores) for sector, scores in groups.items()}


def sector_vote_and_reason(sector: str, sector_averages: dict[str, float]):
    """セクター平均スコアから -1/0/+1 のセクターボーナスと理由テキストを返す。"""
    avg = sector_averages.get(sector)
    if avg is None:
        return 0, None
    if avg >= SECTOR_VOTE_THRESHOLD:
        return 1, f"{sector}セクター全体が本日は堅調（平均スコア+{avg:.1f}）"
    if avg <= -SECTOR_VOTE_THRESHOLD:
        return -1, f"{sector}セクター全体が本日は軟調（平均スコア{avg:.1f}）"
    return 0, None


def load_stock_universe() -> list[dict]:
    """data/universe_prime.json（東証プライム全銘柄）があればそれを使い、
    無ければ組み込みの FALLBACK_STOCKS を使う。
    """
    if UNIVERSE_PATH.exists():
        try:
            cached = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
            stocks = cached.get("stocks", [])
            if stocks:
                print(
                    f"Loaded {len(stocks)} tickers from {UNIVERSE_PATH} "
                    f"(fetched_at={cached.get('fetched_at')})"
                )
                return stocks
        except Exception as exc:
            print(f"[WARN] Failed to load {UNIVERSE_PATH}: {exc}")

    print(f"Using built-in fallback list ({len(FALLBACK_STOCKS)} tickers).")
    return FALLBACK_STOCKS


def download_history_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """複数銘柄の株価を yf.download でまとめて取得する。

    銘柄数が多い場合に Yahoo Finance へ大量の同時リクエストが集中すると
    レート制限にかかりやすいため、CHUNK_SIZE件ずつに分割し、チャンク間に
    短い待機を挟みながら取得する。戻り値は {証券コード: 株価DataFrame} の辞書。
    """
    result: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        chunk_no = i // CHUNK_SIZE + 1
        total_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"Fetching chunk {chunk_no}/{total_chunks} ({len(chunk)} tickers)...")

        try:
            data = yf.download(
                tickers=chunk,
                period=FETCH_PERIOD,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"[WARN] chunk {chunk_no} fetch failed: {exc}")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            available = set(data.columns.get_level_values(0))
            for code in chunk:
                if code not in available:
                    continue
                sub = data[code].dropna(subset=["Close"])
                if not sub.empty:
                    result[code] = sub
        elif len(chunk) == 1:
            sub = data.dropna(subset=["Close"])
            if not sub.empty:
                result[chunk[0]] = sub

        if i + CHUNK_SIZE < len(tickers):
            time.sleep(CHUNK_DELAY_SEC)

    return result


def build_stock_row(entry: dict, history: pd.DataFrame | None) -> dict | None:
    code = entry["code"]
    if history is None or len(history) < 2:
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

    up_probability, expected_return_pct, sample_size, composite_score, bucket = backtest_up_probability(df)
    composite_score_int = None if bucket is None else int(composite_score)
    composite_label = SCORE_LABELS[bucket] if bucket is not None else "データ不足"
    reasons = build_reasons(df.iloc[-1])
    target_price = round_or_none(price * (1 + expected_return_pct / 100), 1) if expected_return_pct is not None else None
    spark = [round_or_none(v, 1) for v in close.tail(SPARKLINE_DAYS).tolist()]

    return {
        "code": code.replace(".T", ""),
        "ticker": code,
        "name": entry["name"],
        "market": entry["market"],
        "sector": entry.get("sector") or "その他",
        "price": round_or_none(price),
        "change_pct": round_or_none(change_pct),
        "rsi": round_or_none(rsi, 1),
        "ma25_dev_pct": round_or_none(ma_dev_pct),
        "signal": build_signal(rsi if rsi is not None else float("nan")),
        "macd_hist": round_or_none(df["macd_hist"].iloc[-1], 2),
        "bb_percent_b": round_or_none(df["bb_percent_b"].iloc[-1] * 100, 1) if pd.notna(df["bb_percent_b"].iloc[-1]) else None,
        "volume_ratio": round_or_none(df["volume_ratio"].iloc[-1], 2),
        "stoch_k": round_or_none(df["stoch_k"].iloc[-1], 1),
        "composite_score": composite_score_int,
        "composite_label": composite_label,
        "up_probability": round_or_none(up_probability, 1),
        "up_probability_samples": sample_size,
        "forward_days": FORWARD_DAYS,
        "reasons": reasons,
        "target_price": target_price,
        "expected_return_pct": round_or_none(expected_return_pct, 1),
        "spark": spark,
    }


def main() -> None:
    entries = load_stock_universe()
    codes = [entry["code"] for entry in entries]

    history_map = download_history_batch(codes)
    print(f"Fetched history for {len(history_map)}/{len(codes)} tickers.")

    results = []
    for entry in entries:
        row = build_stock_row(entry, history_map.get(entry["code"]))
        if row is not None:
            results.append(row)
        else:
            print(f"[WARN] {entry['code']}: no usable data, skipped")
    print(f"Scanned {len(results)} tickers with usable data.")

    print(f"Fetching market index ({MARKET_INDEX_TICKER}) for market-wide context...")
    market_history = download_history_batch([MARKET_INDEX_TICKER]).get(MARKET_INDEX_TICKER)
    if market_history is not None and len(market_history) >= MA_LONG:
        market_vote, market_reason = compute_market_context(market_history)
    else:
        market_vote, market_reason = 0, "日経平均のデータ取得に失敗したため市場全体の判定なし"
    print(f"Market-wide vote: {market_vote} ({market_reason})")

    # セクター全体の地合い（本日スキャンした同一セクター銘柄の平均スコア）を
    # 個別銘柄の総合スコアに反映する。
    sector_averages = compute_sector_averages(results)
    for row in results:
        if row["composite_score"] is None:
            continue
        sector_vote, sector_reason = sector_vote_and_reason(row["sector"], sector_averages)
        row["composite_score"] = row["composite_score"] + market_vote + sector_vote
        if market_vote != 0:
            row["reasons"].append(market_reason)
        if sector_reason:
            row["reasons"].append(sector_reason)

    print("Selecting top candidates...")
    top_candidates = select_top_candidates(results, TOP_N)
    print(f"Selected {len(top_candidates)} top candidates (target {TOP_N * 2}).")

    jst = timezone(timedelta(hours=9))
    payload = {
        "updated_at": datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "count": len(top_candidates),
        "scanned_count": len(results),
        "market_vote": market_vote,
        "market_reason": market_reason,
        "forward_days": FORWARD_DAYS,
        "stocks": top_candidates,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(top_candidates)} stocks -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
