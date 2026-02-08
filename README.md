# CorrelationCalc

Hourly return-correlation and minimum-variance hedge-ratio calculator for **BTC / ETH**, with an auto-refreshing web dashboard.

## Features

- **Hourly correlation** between BTC and ETH returns over configurable lookback windows (30 / 60 / 120 / 150 / 180 days)
- **Minimum-variance hedge ratios** — how many units of the hedge asset to short per unit of the target asset
- **Dollar-neutral weights** that sum to 1 in absolute terms
- **Example position sizing** with an adjustable notional multiplier
- **Multiple data providers** — Binance, CoinGecko, and CryptoCompare with automatic fallback
- **Static web dashboard** served from `web/index.html` and powered by a single JSON file
- **GitHub Actions workflow** that recomputes metrics every 8 hours and commits the result

## Quick start

### Prerequisites

- Python 3.11+
- `pandas` and `requests`

```bash
pip install pandas requests
```

### CLI usage

```bash
# Default: 30-day lookback, auto provider (Binance → CoinGecko → CryptoCompare)
python hedge_metrics.py

# Multiple lookback windows, output to JSON
python hedge_metrics.py --lookbacks 30,60,120,180 --json web/data/latest.json

# Choose a specific data provider
python hedge_metrics.py --provider coingecko

# Custom notional for sizing examples
python hedge_metrics.py --notional 1000

# Skip terminal output and only write JSON
python hedge_metrics.py --json web/data/latest.json --no-print
```

#### CLI options

| Flag | Description |
|---|---|
| `--lookback-days N` | Single lookback window in days (default `30`) |
| `--lookbacks LIST` | Comma-separated lookback windows (overrides `--lookback-days`) |
| `--print-lookback N` | Which lookback to print to stdout (defaults to the first) |
| `--provider` | `auto` (default), `binance`, `coingecko`, or `cryptocompare` |
| `--notional N` | Total absolute notional for sizing examples (default `100`) |
| `--json PATH` | Write full results to a JSON file |
| `--no-print` | Suppress stdout output |

## Web dashboard

The file `web/index.html` is a self-contained, single-page dashboard that reads `web/data/latest.json` and displays:

- Correlation value and lookback selector
- Hourly return statistics (mean, std, min, max)
- Hedge-ratio table
- Interactive position-sizing examples with a notional multiplier slider

Deploy the `web/` directory to any static host (e.g. Vercel, GitHub Pages, Netlify).

## Automated refresh

A GitHub Actions workflow (`.github/workflows/refresh-metrics.yml`) runs every 8 hours:

1. Checks out the repository
2. Installs Python and dependencies
3. Runs `hedge_metrics.py` with all five lookback windows
4. Commits the updated `web/data/latest.json` back to `main`

The workflow can also be triggered manually via **Actions → Run workflow** or on every push to `main`.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `COINGECKO_API_KEY` | No | CoinGecko Pro API key (uses demo key if not set) |
| `CRYPTOCOMPARE_API_KEY` | No | CryptoCompare API key for higher rate limits |

## Project structure

```
├── hedge_metrics.py               # Core computation and CLI
├── web/
│   ├── index.html                 # Static dashboard
│   ├── logo.svg                   # App icon
│   └── data/
│       └── latest.json            # Auto-generated metrics (committed by CI)
└── .github/
    └── workflows/
        └── refresh-metrics.yml    # Scheduled refresh workflow
```

## License

This project does not currently include a license file. All rights are reserved by the author unless otherwise stated.
