# Stochastic FIRE Simulator

A Monte Carlo simulator for FIRE (Financial Independence / Retire Early) withdrawal risk.

It estimates how many years it takes until portfolio balance reaches zero, using either:

- `normal` model: random yearly market/inflation draws from normal distributions
- `block-bootstrap` model: random multi-year historical blocks of `(market return, inflation)`

100% vibe-coded (Using Codex 5.3).

## What It Outputs

Each run prints:

- mean and standard deviation of years-to-zero
- survival horizons (95% down to 0%, in 5% steps, stops when horizon reaches cap)
- ruin/survival percentages at key horizons
- a survival scatter plot file:
  - `survival_scatter.png` when `matplotlib` is available
  - `survival_scatter.svg` as fallback

The scatter plot uses:

- x-axis: survival to year
- y-axis: survival probability (%)
- range: 0..100 years
- points: 100

---

## Requirements

- Python 3.10+ (recommended)
- Optional: `matplotlib` for PNG plot output

If `matplotlib` is missing, the simulator still runs and writes an SVG plot instead.

---

## Quick Start

Run from the project root:

```bash
python3 fire_simulator.py --model normal --initial-money 1m --withdrawal-amount 80k --preset total-market
```

Money arguments support suffixes:

- `k` = thousand (e.g. `80k`)
- `m` = million (e.g. `1m`)

Also accepted: plain numbers, comma/underscore separators (e.g. `1,000,000`, `1_000_000`).

---

## Main Examples

### 1) Normal model with preset assumptions

```bash
python3 fire_simulator.py \
  --model normal \
  --initial-money 1m \
  --withdrawal-amount 80k \
  --preset total-market \
  --simulations 1000 \
  --seed 42
```

### 2) Normal model with manual assumptions

```bash
python3 fire_simulator.py \
  --model normal \
  --initial-money 1m \
  --withdrawal-amount 80k \
  --market-mu 0.10 \
  --market-sigma 0.18 \
  --inflation-mu 0.025 \
  --inflation-sigma 0.015 \
  --simulations 1000
```

### 3) Block-bootstrap with US total-market history

```bash
python3 fire_simulator.py \
  --model block-bootstrap \
  --initial-money 1m \
  --withdrawal-amount 80k \
  --market-series total-market \
  --block-years 5 \
  --simulations 1000 \
  --seed 42
```

### 4) Block-bootstrap with custom historical window

```bash
python3 fire_simulator.py \
  --model block-bootstrap \
  --initial-money 1m \
  --withdrawal-amount 80k \
  --market-series sp500 \
  --historical-start-year 1990 \
  --historical-end-year 2020 \
  --simulations 1000
```

---

## Important Model Notes

- Withdrawal is inflation-adjusted each year.
- Simulation horizon is hard-capped at 500 years.
  - If `--max-years` is above 500, it is capped and a note is printed.
- Survival levels default to `95%, 90%, ..., 5%, 0%`.
- In `block-bootstrap`, historical data is fetched at runtime from online sources.

---

## CLI Arguments (Common)

- `--model {normal,block-bootstrap}`
- `--initial-money <money>`
- `--withdrawal-amount <money>`
- `--simulations <int>` (default: `1000`)
- `--max-years <int>` (default: `500`, capped at `500`)
- `--survival-probabilities <float ...>` (defaults: `0.95` to `0.0` in `0.05` steps)
- `--seed <int>`

### Normal-only

- `--preset {sp500,total-market}`
- `--market-mu <float>`
- `--market-sigma <float>`
- `--inflation-mu <float>`
- `--inflation-sigma <float>`

### Block-bootstrap-only

- `--market-series {sp500,total-market}`
- `--block-years <int>` (default: `5`)
- `--historical-start-year <int>`
- `--historical-end-year <int>`

---

## Historical Data Sources (Bootstrap)

- S&P 500 total return: Damodaran historical returns table
- US total-market proxy: Kenneth French factors (market + risk-free)
- US inflation: FRED CPI (`CPIAUCSL`)

