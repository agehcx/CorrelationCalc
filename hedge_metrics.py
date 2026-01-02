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
    python hedge_metrics.py
"""



BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
CRYPTOCOMPARE_HISTO = "https://min-api.cryptocompare.com/data/v2/histohour"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"
LOOKBACK_DAYS = 30  # default single lookback
LOOKBACK_CHOICES = [30, 60, 120, 180]
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


def fetch_coingecko(symbol: str, start_ts: int, end_ts: int, lookback_days: int) -> pd.DataFrame:
    """Fetch prices from CoinGecko market_chart/range to allow >90d hourly.

    Requires COINGECKO_API_KEY (pro) or falls back to demo key header.
    """
    coin_id = COINGECKO_IDS[symbol]
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": int(start_ts / 1000),
        "to": int(end_ts / 1000),
        "precision": "full",
    }

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

    resp = requests.get(url, params=params, headers=headers, timeout=30)
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


def fetch_cryptocompare(symbol: str, start_ts: int, end_ts: int, lookback_days: int) -> pd.DataFrame:
    """Fetch hourly prices from CryptoCompare histohour with simple backfill pagination."""
    cc_symbol = symbol.replace("USDT", "")  # BTCUSDT -> BTC
    frames: List[pd.DataFrame] = []
    current_end = end_ts
    api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
    headers = {"accept": "application/json", "User-Agent": "correlation-calc/1.0"}
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"

    while current_end > start_ts:
        params = {
            "fsym": cc_symbol,
            "tsym": "USD",
            "limit": 2000,
            "toTs": int(current_end / 1000),
        }
        resp = requests.get(CRYPTOCOMPARE_HISTO, params=params, headers=headers, timeout=20)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"CryptoCompare fetch failed for {symbol}: {e}") from e

        payload = resp.json().get("Data", {}).get("Data", [])
        if not payload:
            break
        df = pd.DataFrame(payload)[["time", "close"]]
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        frames.append(df[["timestamp", "close"]])

        oldest_ms = int(df["timestamp"].min().timestamp() * 1000)
        # step back one hour before oldest to continue pagination
        current_end = oldest_ms - 3600 * 1000
        # respect rate limits
        time.sleep(0.15)

        if oldest_ms <= start_ts:
            break

    if not frames:
        raise RuntimeError(f"No data fetched from CryptoCompare for {symbol}")

    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    out = out[out["timestamp"] >= pd.to_datetime(start_ts, unit="ms", utc=True)]
    return out[["timestamp", "close"]]


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
        return fetch_coingecko(symbol, start_ts, end_ts, lookback_days)
    if provider == "binance":
        return fetch_klines(symbol, start_ts, end_ts)
    if provider == "cryptocompare":
        return fetch_cryptocompare(symbol, start_ts, end_ts, lookback_days)

    # auto
    try:
        return fetch_klines(symbol, start_ts, end_ts)
    except Exception as e:
        print(f"[warn] Binance failed for {symbol}: {e}. Falling back to CoinGecko...", file=sys.stderr)
        try:
            return fetch_coingecko(symbol, start_ts, end_ts, lookback_days)
        except Exception as e2:
            print(f"[warn] CoinGecko failed for {symbol}: {e2}. Falling back to CryptoCompare...", file=sys.stderr)
            return fetch_cryptocompare(symbol, start_ts, end_ts, lookback_days)


def hedge_ratio_min_var(corr_val: float, sigma_target: float, sigma_hedge: float) -> float:
    return corr_val * (sigma_target / sigma_hedge)


def dollar_neutral_weights(h: float) -> tuple[float, float]:
    """Return weights (target, hedge) that sum to 1 in absolute dollars."""
    total = 1 + abs(h)
    return 1 / total, -abs(h) / total


def load_prices(max_lookback_days: int, provider: str) -> tuple[pd.DataFrame, dt.datetime]:
    """Fetch hourly prices for the maximum lookback window once and return tidy frame."""
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=max_lookback_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    prices = {}
    for sym in SYMBOLS:
        prices[sym] = fetch_prices(sym, start_ms, end_ms, max_lookback_days, provider=provider).set_index("timestamp")

    df = pd.concat(prices.values(), axis=1, keys=SYMBOLS)
    df = df.dropna()
    df.columns = df.columns.droplevel(1)
    df = df.rename(columns={"BTCUSDT": "btc", "ETHUSDT": "eth"})
    return df, end


def compute_single_view(
    price_df: pd.DataFrame,
    end: dt.datetime,
    lookback_days: int,
    notional: float,
    provider: str,
) -> dict:
    cutoff = end - dt.timedelta(days=lookback_days)
    window = price_df[price_df.index >= cutoff]
    if window.empty:
        raise RuntimeError(f"No price data available for {lookback_days}d window")

    returns = window.pct_change().dropna()
    if returns.empty:
        raise RuntimeError(f"No returns for {lookback_days}d window")

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
        "lookback_days": lookback_days,
        "corr": corr,
        "stats": stats.to_dict(),
        "hedge_table": table.to_dict(orient="records"),
        "examples": examples,
    }


def compute_all_metrics(
    lookbacks: List[int],
    notional: float = 100.0,
    provider: str = "auto",
) -> dict:
    """Compute metrics for multiple lookbacks using a single price fetch."""

    if not lookbacks:
        raise ValueError("lookbacks list cannot be empty")

    max_lb = max(lookbacks)
    price_df, end = load_prices(max_lb, provider)

    views = {}
    for lb in sorted(lookbacks):
        views[str(lb)] = compute_single_view(price_df, end, lb, notional, provider)

    return {
        "generated_at": end.isoformat() + "Z",
        "lookbacks": views,
        "example_notional": notional,
        "provider": provider,
    }


def print_metrics(view: dict, notional: float, provider: str, generated_at: str | None = None) -> None:
    corr = view["corr"]
    stats = pd.DataFrame(view["stats"])
    stats = stats[["mean", "std", "min", "max"]]

    print(f"Hourly return correlation (BTC vs ETH) over ~{view['lookback_days']}d: {corr:.4f}")
    if generated_at:
        print(f"Generated at: {generated_at} | Provider: {provider}")

    print("\nStats:")
    print(stats.T)

    table = pd.DataFrame(view["hedge_table"])
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
    for ex in view["examples"]:
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
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS, help="Single lookback window in days (legacy)")
    parser.add_argument(
        "--lookbacks",
        type=str,
        default=None,
        help="Comma-separated lookbacks in days (e.g., 30,60,120,180). Overrides --lookback-days if provided.",
    )
    parser.add_argument(
        "--print-lookback",
        type=int,
        default=None,
        help="Which lookback to print to stdout (defaults to first provided lookback)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "binance", "coingecko", "cryptocompare"],
        default="auto",
        help="Data source: auto (binance -> coingecko -> cryptocompare), or choose binance/coingecko/cryptocompare",
    )
    parser.add_argument("--no-print", action="store_true", help="Skip stdout printing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lookbacks:
        lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    else:
        lookbacks = [args.lookback_days]

    metrics = compute_all_metrics(lookbacks=lookbacks, notional=args.notional, provider=args.provider)

    if not args.no_print:
        to_print = str(args.print_lookback or lookbacks[0])
        if to_print not in metrics["lookbacks"]:
            raise ValueError(f"Requested print lookback {to_print} not in computed set {list(metrics['lookbacks'].keys())}")
        print_metrics(metrics["lookbacks"][to_print], notional=args.notional, provider=metrics["provider"], generated_at=metrics["generated_at"])

    if args.json:
        write_json(metrics, args.json)


if __name__ == "__main__":
    main()