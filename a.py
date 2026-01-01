from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import List
import pandas as pd
import requests

"""
Compute hourly return correlation (ETHUSDT vs BTCUSDT) over the last ~6 months
using Binance spot klines (no API key required).

Run:
    pyv a.py
"""



BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"
LOOKBACK_DAYS = 180  # ~6 months
LIMIT = 1000  # Binance max per request


def fetch_klines(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Paginate Binance klines from start_ts to end_ts (both ms)."""
    frames: List[pd.DataFrame] = []
    current = start_ts
    while current < end_ts:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current,
            "endTime": end_ts,
            "limit": LIMIT,
        }
        resp = requests.get(BINANCE_KLINES, params=params, timeout=10)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Propagate with context so callers can decide to fall back.
            raise RuntimeError(f"Binance klines fetch failed for {symbol}: {e}") from e
        data = resp.json()
        if not data:
            break

        df = pd.DataFrame(
            data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "qav",
                "num_trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
        frames.append(df[["open_time", "close"]].astype({"open_time": "int64", "close": "float"}))

        # next page: last close_time + 1 ms
        last_close_time = int(data[-1][6])
        current = last_close_time + 1

        # avoid hitting rate limits
        time.sleep(0.05)

    if not frames:
        raise RuntimeError(f"No data fetched for {symbol}")

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    return out[["timestamp", "close"]]


COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
}


def fetch_coingecko(symbol: str, lookback_days: int) -> pd.DataFrame:
    """Fetch hourly prices from CoinGecko market_chart as a fallback.

    CoinGecko now requires an API key. Provide via env COINGECKO_API_KEY.
    If missing, we try the public demo key but it may be rate-limited.
    """
    coin_id = COINGECKO_IDS[symbol]
    url = COINGECKO_MARKET_CHART.format(id=coin_id)
    params = {"vs_currency": "usd", "days": lookback_days, "interval": "hourly"}

    api_key = os.getenv("COINGECKO_API_KEY")
    demo_key = "CG-DATA-API-KEY"
    headers = {
        "accept": "application/json",
        "User-Agent": "correlation-calc/1.0",
    }
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
    else:
        headers["x-cg-demo-api-key"] = demo_key

    resp = requests.get(url, params=params, headers=headers, timeout=20)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"CoinGecko fetch failed for {symbol}: {e}") from e
    data = resp.json().get("prices", [])
    if not data:
        raise RuntimeError(f"CoinGecko returned no data for {symbol}")
    df = pd.DataFrame(data, columns=["open_time", "close"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["timestamp", "close"]]


def fetch_prices(
    symbol: str,
    start_ts: int,
    end_ts: int,
    lookback_days: int,
    provider: str = "auto",
) -> pd.DataFrame:
    """Fetch prices using provider policy.

    provider options:
      - "auto" (default): try Binance, fall back to CoinGecko
      - "binance": use Binance only
      - "coingecko": use CoinGecko only
    """

    if provider == "coingecko":
        return fetch_coingecko(symbol, lookback_days)
    if provider == "binance":
        return fetch_klines(symbol, start_ts, end_ts)

    # auto
    try:
        return fetch_klines(symbol, start_ts, end_ts)
    except Exception as e:
        print(f"[warn] Binance failed for {symbol}: {e}. Falling back to CoinGecko...", file=sys.stderr)
        return fetch_coingecko(symbol, lookback_days)


def hedge_ratio_min_var(corr_val: float, sigma_target: float, sigma_hedge: float) -> float:
    return corr_val * (sigma_target / sigma_hedge)


def dollar_neutral_weights(h: float) -> tuple[float, float]:
    """Return weights (target, hedge) that sum to 1 in absolute dollars."""
    total = 1 + abs(h)
    return 1 / total, -abs(h) / total


def compute_metrics(
    lookback_days: int = LOOKBACK_DAYS,
    notional: float = 100.0,
    provider: str = "auto",
) -> dict:
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    prices = {}
    for sym in SYMBOLS:
        prices[sym] = fetch_prices(sym, start_ms, end_ms, lookback_days, provider=provider).set_index("timestamp")

    df = pd.concat(prices.values(), axis=1, keys=SYMBOLS)
    df = df.dropna()
    df.columns = df.columns.droplevel(1)
    df = df.rename(columns={"BTCUSDT": "btc", "ETHUSDT": "eth"})

    returns = df.apply(lambda x: x.pct_change()).dropna()
    corr = float(returns["btc"].corr(returns["eth"]))

    stats = returns.describe().T[["mean", "std", "min", "max"]]
    std_btc = float(stats.loc["btc", "std"])
    std_eth = float(stats.loc["eth", "std"])

    h_btc_hedged_with_eth = hedge_ratio_min_var(corr, std_btc, std_eth)
    h_eth_hedged_with_btc = hedge_ratio_min_var(corr, std_eth, std_btc)

    w_btc, w_eth = dollar_neutral_weights(h_btc_hedged_with_eth)
    w_eth_as_target, w_btc_as_hedge = dollar_neutral_weights(h_eth_hedged_with_btc)

    rows = [
        {
            "target": "BTC",
            "hedge": "ETH",
            "hedge_ratio_units": h_btc_hedged_with_eth,
            "target_w": w_btc,
            "hedge_w": w_eth,
        },
        {
            "target": "ETH",
            "hedge": "BTC",
            "hedge_ratio_units": h_eth_hedged_with_btc,
            "target_w": w_eth_as_target,
            "hedge_w": w_btc_as_hedge,
        },
    ]

    table = pd.DataFrame(rows)

    examples = []
    hedge_sign = -1 if corr >= 0 else 1  # positive corr -> short hedge
    for row in rows:
        tgt = row["target"]
        hed = row["hedge"]
        tgt_dollars = row["target_w"] * notional
        hed_dollars = hedge_sign * abs(row["hedge_w"]) * notional
        hedge_units = hedge_sign * abs(row["hedge_ratio_units"])
        examples.append(
            {
                "target": tgt,
                "hedge": hed,
                "target_dollars": tgt_dollars,
                "hedge_dollars": hed_dollars,
                "hedge_units_per_target_unit": hedge_units,
            }
        )

    return {
        "generated_at": end.isoformat() + "Z",
        "lookback_days": lookback_days,
        "example_notional": notional,
        "provider": provider,
        "corr": corr,
        "stats": stats.to_dict(),
        "hedge_table": table.to_dict(orient="records"),
        "examples": examples,
    }


def print_metrics(metrics: dict, notional: float) -> None:
    corr = metrics["corr"]
    stats = pd.DataFrame(metrics["stats"])
    stats = stats[["mean", "std", "min", "max"]]

    print(f"Hourly return correlation (BTC vs ETH) over ~{metrics['lookback_days']}d: {corr:.4f}")

    print("\nStats:")
    print(stats.T)

    table = pd.DataFrame(metrics["hedge_table"])
    table_display = table.copy()
    table_display["hedge_ratio_units"] = table_display["hedge_ratio_units"].map(lambda x: f"{x:+.3f}")
    table_display["target_w"] = table_display["target_w"].map(lambda x: f"{x:+.3f}")
    table_display["hedge_w"] = table_display["hedge_w"].map(lambda x: f"{x:+.3f}")
    table_display = table_display.rename(columns={"hedge_ratio_units": "hedge_ratio (units)"})

    print("\nSuggested min-var hedge ratios (long target, short hedge):")
    print(table_display.to_string(index=False))
    print("\nColumns:")
    print("- hedge_ratio (units): short this many hedge units per 1 unit long target")
    print("- target_w / hedge_w: dollar-neutral weights (abs weights sum to 1)")

    print(f"\nExample sizing with ${notional} total abs notional (long target, short hedge):")
    for ex in metrics["examples"]:
        tgt = ex["target"]
        hed = ex["hedge"]
        hedge_units = ex["hedge_units_per_target_unit"]
        sign = "short" if hedge_units < 0 else "long"
        print(
            f"- {tgt} as target: long ${ex['target_dollars']:.2f} {tgt}, {sign} ${abs(ex['hedge_dollars']):.2f} {hed} "
            f"(per 1 {tgt} unit, {sign} {abs(hedge_units):.3f} {hed})"
        )


def write_json(metrics: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC/ETH hedge ratios and stats")
    parser.add_argument("--json", help="Path to write JSON output", default=None)
    parser.add_argument("--notional", type=float, default=100.0, help="Example total absolute notional for sizing output")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS, help="Lookback window in days")
    parser.add_argument(
        "--provider",
        choices=["auto", "binance", "coingecko"],
        default="auto",
        help="Data source: auto (try Binance, fallback CoinGecko), binance, or coingecko",
    )
    parser.add_argument("--no-print", action="store_true", help="Skip stdout printing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(lookback_days=args.lookback_days, notional=args.notional, provider=args.provider)

    if not args.no_print:
        print_metrics(metrics, notional=args.notional)

    if args.json:
        write_json(metrics, args.json)


if __name__ == "__main__":
    main()