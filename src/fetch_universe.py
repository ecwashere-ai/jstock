#!/usr/bin/env python3
"""東証プライム市場の上場銘柄一覧を JPX公式サイトから取得し、
data/universe_prime.json に保存するスクリプト。

JPXは上場銘柄一覧（実質的な公式データ）を随時更新しているため、
このスクリプトはキャッシュファイルが UNIVERSE_MAX_AGE_DAYS より古い
場合にのみ再取得する（update_data.yml から毎回呼ばれる想定）。

取得・パースに失敗した場合は既存のキャッシュファイルを変更せずに終了する。
これにより、JPX側のファイル形式変更や一時的な障害があっても、
日次の株価更新（update_stocks.py）が完全に止まることを防ぐ。
"""

import io
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

# JPX「その他統計資料」ページで公開されている上場銘柄一覧(Excel)。
# JPXはこのファイルを定期的に内容だけ更新しているため URL 自体は
# 長期的に安定しているが、将来 JPX 側でパスが変更される可能性はある。
# その場合はこの定数を JPX サイトで確認した最新の URL に更新すること。
JPX_LISTED_ISSUES_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

UNIVERSE_MAX_AGE_DAYS = 7
MIN_EXPECTED_PRIME_COUNT = 500  # 東証プライムは通常1,000銘柄超のため、これを下回ったらパース失敗とみなす

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "universe_prime.json"


def is_cache_fresh() -> bool:
    if not OUTPUT_PATH.exists():
        return False
    try:
        cached = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(cached["fetched_at"])
    except Exception:
        return False
    return datetime.now(timezone.utc) - fetched_at < timedelta(days=UNIVERSE_MAX_AGE_DAYS)


def fetch_prime_universe() -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; jstock-universe-fetcher/1.0)"}
    resp = requests.get(JPX_LISTED_ISSUES_URL, headers=headers, timeout=60)
    resp.raise_for_status()

    df = pd.read_excel(io.BytesIO(resp.content))
    df.columns = [str(c).strip() for c in df.columns]

    required_columns = {"コード", "銘柄名", "市場・商品区分", "33業種区分"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"想定した列が見つかりません: {missing}（取得した列: {list(df.columns)}）")

    market = df["市場・商品区分"].astype(str)
    is_prime_stock = market.str.contains("プライム") & market.str.contains("内国株式")
    prime_df = df[is_prime_stock]

    universe = []
    for _, row in prime_df.iterrows():
        code = str(row["コード"]).strip()
        name = str(row["銘柄名"]).strip()
        sector = str(row["33業種区分"]).strip()
        if not code or not code[0].isdigit():
            continue
        universe.append({
            "code": f"{code}.T",
            "name": name,
            "market": "東証プライム",
            "sector": sector or "その他",
        })

    return universe


def main() -> None:
    force = "--force" in sys.argv
    if not force and is_cache_fresh():
        print(f"{OUTPUT_PATH} is fresh (< {UNIVERSE_MAX_AGE_DAYS} days old). Skipping fetch.")
        return

    try:
        universe = fetch_prime_universe()
    except Exception as exc:
        print(f"[ERROR] Failed to fetch/parse JPX listed issues: {exc}")
        if OUTPUT_PATH.exists():
            print("既存のキャッシュファイルを維持します。")
            return
        raise

    if len(universe) < MIN_EXPECTED_PRIME_COUNT:
        print(
            f"[ERROR] Parsed only {len(universe)} tickers "
            f"(expected >= {MIN_EXPECTED_PRIME_COUNT}); likely a parsing issue. "
            "Aborting without overwriting the cache."
        )
        sys.exit(1)

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(universe),
        "source": JPX_LISTED_ISSUES_URL,
        "stocks": universe,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(universe)} Prime-market tickers -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
