"""Monte Carlo simulator for FIRE portfolio depletion.

Inputs are annualized values. Two models are supported:
- normal: draws market and inflation from normal(mu, sigma)
- block-bootstrap: samples historical multi-year blocks of
  (market return, inflation) pairs

Normal-model inputs:
- current money amount (e.g. 1m)
- withdrawal amount in money units (e.g. 80k)
- inflation normal distribution (mu, sigma)
- market growth normal distribution (mu, sigma)
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import re
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Historical calibration presets (annual nominal returns, mu/sigma for normal draws):
# - S&P 500 total return (Damodaran, 1928-2025): mu=0.1186, sigma=0.1940
# - Vanguard total-market proxy (Fama-French US market, 1996-2021): mu=0.1193, sigma=0.1788
#   Use this as a proxy for VTI/VTSAX (US total market), not S&P 500-only funds.
# - US inflation CPI (FRED CPIAUCSL, 1948-2025): mu=0.0352, sigma=0.0277
# - US inflation CPI (FRED CPIAUCSL, 1996-2021): mu=0.0224, sigma=0.0109
PRESETS: dict[str, dict[str, float]] = {
  "sp500": {
    "market_mu": 0.1186,
    "market_sigma": 0.1940,
    "inflation_mu": 0.0352,
    "inflation_sigma": 0.0277,
  },
  "total-market": {
    "market_mu": 0.1193,
    "market_sigma": 0.1788,
    "inflation_mu": 0.0224,
    "inflation_sigma": 0.0109,
  },
}

# Data sources used for block-bootstrap historical tables.
SP500_TOTAL_RETURN_URL = (
  "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html"
)
FF_MARKET_FACTORS_URL = (
  "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
  "ftp/F-F_Research_Data_Factors.CSV"
)
US_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
HARD_MAX_YEARS = 500
DEFAULT_SURVIVAL_PROBABILITIES = [
  probability / 100.0 for probability in range(95, -5, -5)
]
DEFAULT_PLOT_MAX_YEAR = 100.0
DEFAULT_PLOT_POINTS = 100
DEFAULT_PLOT_OUTPUT = "survival_scatter.png"


def parse_money(value: str) -> float:
  text = value.strip().lower().replace(",", "").replace("_", "")
  match = re.fullmatch(r"([0-9]*\.?[0-9]+)([km]?)", text)
  if match is None:
    raise argparse.ArgumentTypeError(
      "Invalid money value. Use a number with optional k/m suffix, e.g. 80k, 1m."
    )
  amount = float(match.group(1))
  suffix = match.group(2)
  multiplier = 1.0
  if suffix == "k":
    multiplier = 1_000.0
  elif suffix == "m":
    multiplier = 1_000_000.0
  return amount * multiplier


@dataclass(frozen=True)
class SimulationResult:
  mean_years_to_zero: float
  std_years_to_zero: float
  years_to_zero: list[int]
  survival_horizons: dict[float, int]
  simulations: int
  max_years: int


def _clamp_rate(sampled_rate: float) -> float:
  """Avoid impossible values below -100% from normal draws."""
  return max(sampled_rate, -0.9999)


def _simulate_one_path(
  initial_money: float,
  withdrawal_amount: float,
  inflation_mu: float,
  inflation_sigma: float,
  market_mu: float,
  market_sigma: float,
  rng: random.Random,
  max_years: int,
) -> int:
  balance = initial_money
  annual_withdrawal = withdrawal_amount

  for year in range(1, max_years + 1):
    market_return = _clamp_rate(rng.gauss(market_mu, market_sigma))
    balance *= 1.0 + market_return
    balance -= annual_withdrawal

    if balance <= 0.0:
      return year

    inflation = _clamp_rate(rng.gauss(inflation_mu, inflation_sigma))
    annual_withdrawal *= 1.0 + inflation

  return max_years


def _compute_survival_horizons(
  years_to_zero: list[int],
  survival_probabilities: tuple[float, ...],
  max_years: int,
) -> dict[float, int]:
  sorted_desc = sorted(years_to_zero, reverse=True)
  sample_size = len(sorted_desc)
  horizons: dict[float, int] = {}

  for survival_probability in survival_probabilities:
    if not 0.0 <= survival_probability <= 1.0:
      raise ValueError("survival_probabilities values must be in [0, 1]")
    if survival_probability == 0.0:
      horizons[survival_probability] = max_years
      continue
    rank = math.ceil(survival_probability * sample_size)
    horizons[survival_probability] = sorted_desc[rank - 1]

  return horizons


def _validate_common_inputs(
  initial_money: float,
  withdrawal_amount: float,
  simulations: int,
  max_years: int,
) -> None:
  if initial_money <= 0.0:
    raise ValueError("initial_money must be > 0")
  if withdrawal_amount < 0.0:
    raise ValueError("withdrawal_amount must be >= 0")
  if simulations <= 0:
    raise ValueError("simulations must be > 0")
  if max_years <= 0:
    raise ValueError("max_years must be > 0")


def _build_simulation_result(
  years_to_zero: list[int],
  simulations: int,
  max_years: int,
  survival_probabilities: tuple[float, ...],
) -> SimulationResult:
  mean_years = statistics.fmean(years_to_zero)
  std_years = statistics.stdev(years_to_zero) if simulations > 1 else 0.0
  survival_horizons = _compute_survival_horizons(
    years_to_zero, survival_probabilities, max_years
  )
  return SimulationResult(
    mean_years_to_zero=mean_years,
    std_years_to_zero=std_years,
    years_to_zero=years_to_zero,
    survival_horizons=survival_horizons,
    simulations=simulations,
    max_years=max_years,
  )


def _print_percentage_summary(result: SimulationResult) -> None:
  horizons = (10, 20, 30, 40, 50, 100, 200)
  for horizon in horizons:
    if horizon >= result.max_years:
      continue
    ruined_by_horizon = sum(year <= horizon for year in result.years_to_zero)
    print(f"Ruin by year {horizon}: {ruined_by_horizon / result.simulations:.1%}")

  ruined_before_max = sum(year < result.max_years for year in result.years_to_zero)
  survived_to_max = result.simulations - ruined_before_max
  print(f"Ruin before year {result.max_years}: {ruined_before_max / result.simulations:.1%}")
  print(f"Survival to year {result.max_years}: {survived_to_max / result.simulations:.1%}")
  if survived_to_max > 0:
    print(
      f"Count surviving to year {result.max_years}: "
      f"{survived_to_max}/{result.simulations}"
    )


def _compute_survival_scatter_points(
  years_to_zero: list[int],
  max_year: float,
  points: int,
) -> list[tuple[float, float]]:
  if max_year <= 0.0:
    raise ValueError("plot_max_year must be > 0.")
  if points <= 1:
    raise ValueError("plot_points must be > 1.")
  total_simulations = len(years_to_zero)
  step = max_year / (points - 1)
  scatter_points: list[tuple[float, float]] = []
  for index in range(points):
    year = step * index
    survived_count = sum(path_year >= year for path_year in years_to_zero)
    survived_pct = (survived_count / total_simulations) * 100.0
    scatter_points.append((year, survived_pct))
  return scatter_points


def _format_tick(value: float) -> str:
  rounded = round(value)
  if abs(value - rounded) < 1e-9:
    return str(int(rounded))
  return f"{value:.1f}"


def _write_survival_scatter_svg(
  years_to_zero: list[int],
  output_path: str,
  max_year: float,
  points: int,
) -> str:
  scatter_points = _compute_survival_scatter_points(
    years_to_zero=years_to_zero,
    max_year=max_year,
    points=points,
  )

  width = 960
  height = 540
  margin_left = 80
  margin_right = 25
  margin_top = 35
  margin_bottom = 70
  plot_width = width - margin_left - margin_right
  plot_height = height - margin_top - margin_bottom

  def x_to_px(x_value: float) -> float:
    return margin_left + (x_value / max_year) * plot_width

  def y_to_px(y_value: float) -> float:
    return margin_top + ((100.0 - y_value) / 100.0) * plot_height

  svg_lines = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
    f'viewBox="0 0 {width} {height}">',
    '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
  ]

  tick_count = 5
  for tick_index in range(tick_count + 1):
    x_value = max_year * tick_index / tick_count
    x_pixel = x_to_px(x_value)
    svg_lines.append(
      f'<line x1="{x_pixel:.2f}" y1="{margin_top}" '
      f'x2="{x_pixel:.2f}" y2="{margin_top + plot_height}" '
      'stroke="#e0e0e0" stroke-width="1"/>'
    )
    svg_lines.append(
      f'<text x="{x_pixel:.2f}" y="{height - 35}" text-anchor="middle" '
      'font-size="12" fill="#444">'
      f"{_format_tick(x_value)}"
      "</text>"
    )

  for y_value in range(0, 101, 20):
    y_pixel = y_to_px(float(y_value))
    svg_lines.append(
      f'<line x1="{margin_left}" y1="{y_pixel:.2f}" '
      f'x2="{margin_left + plot_width}" y2="{y_pixel:.2f}" '
      'stroke="#e0e0e0" stroke-width="1"/>'
    )
    svg_lines.append(
      f'<text x="{margin_left - 10}" y="{y_pixel + 4:.2f}" text-anchor="end" '
      'font-size="12" fill="#444">'
      f"{y_value}%"
      "</text>"
    )

  svg_lines.append(
    f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
    f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
    'stroke="#333" stroke-width="2"/>'
  )
  svg_lines.append(
    f'<line x1="{margin_left}" y1="{margin_top}" '
    f'x2="{margin_left}" y2="{margin_top + plot_height}" '
    'stroke="#333" stroke-width="2"/>'
  )

  for x_value, y_value in scatter_points:
    svg_lines.append(
      f'<circle cx="{x_to_px(x_value):.2f}" cy="{y_to_px(y_value):.2f}" '
      'r="2.8" fill="#1f77b4" />'
    )

  svg_lines.append(
    f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 10}" '
    'text-anchor="middle" font-size="14" fill="#222">'
    "Survival to year"
    "</text>"
  )
  svg_lines.append(
    f'<text x="20" y="{margin_top + plot_height / 2:.2f}" '
    'text-anchor="middle" font-size="14" fill="#222" '
    f'transform="rotate(-90 20 {margin_top + plot_height / 2:.2f})">'
    "Survival probability (%)"
    "</text>"
  )
  svg_lines.append(
    f'<text x="{margin_left + plot_width / 2:.2f}" y="22" '
    'text-anchor="middle" font-size="16" fill="#111">'
    "Survival Scatter Plot"
    "</text>"
  )
  svg_lines.append("</svg>")

  path = Path(output_path).expanduser()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(svg_lines), encoding="utf-8")
  return str(path)


def _write_survival_scatter_plot(
  years_to_zero: list[int],
  output_path: str,
  max_year: float = DEFAULT_PLOT_MAX_YEAR,
  points: int = DEFAULT_PLOT_POINTS,
) -> str:
  scatter_points = _compute_survival_scatter_points(
    years_to_zero=years_to_zero,
    max_year=max_year,
    points=points,
  )
  x_values = [x_value for x_value, _ in scatter_points]
  y_values = [y_value for _, y_value in scatter_points]

  path = Path(output_path).expanduser()
  path.parent.mkdir(parents=True, exist_ok=True)

  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except Exception:
    svg_path = path.with_suffix(".svg")
    return _write_survival_scatter_svg(
      years_to_zero=years_to_zero,
      output_path=str(svg_path),
      max_year=max_year,
      points=points,
    )

  figure, axis = plt.subplots(figsize=(10, 6))
  axis.scatter(x_values, y_values, s=18, alpha=0.85, color="#1f77b4")
  axis.set_xlim(0.0, max_year)
  axis.set_ylim(0.0, 100.0)
  axis.set_xlabel("Survival to year")
  axis.set_ylabel("Survival probability (%)")
  axis.set_title("Survival Scatter Plot")
  axis.grid(True, alpha=0.25)
  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)
  return str(path)


def _fetch_text(url: str) -> str:
  try:
    with urllib.request.urlopen(url, timeout=30) as response:
      return response.read().decode("utf-8", errors="ignore")
  except urllib.error.URLError as error:
    raise ValueError(f"Failed to fetch historical data from {url}: {error}") from error


def _strip_html(html_fragment: str) -> str:
  text = re.sub(r"<[^>]+>", "", html_fragment)
  return re.sub(r"\s+", " ", text).strip()


def _load_sp500_total_returns_by_year() -> dict[int, float]:
  html = _fetch_text(SP500_TOTAL_RETURN_URL)
  returns_by_year: dict[int, float] = {}
  table_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.DOTALL | re.IGNORECASE)
  for table_row in table_rows:
    table_cells = re.findall(
      r"<td[^>]*>(.*?)</td>", table_row, flags=re.DOTALL | re.IGNORECASE
    )
    if len(table_cells) < 2:
      continue
    year_text = _strip_html(table_cells[0])
    return_text = _strip_html(table_cells[1])
    if not re.fullmatch(r"\d{4}", year_text):
      continue
    if not re.fullmatch(r"-?\d+\.\d+%", return_text):
      continue
    returns_by_year[int(year_text)] = float(return_text[:-1]) / 100.0
  if not returns_by_year:
    raise ValueError("Could not parse S&P 500 historical returns table.")
  return returns_by_year


def _load_total_market_returns_by_year() -> dict[int, float]:
  csv_text = _fetch_text(FF_MARKET_FACTORS_URL)
  monthly_returns_by_year: dict[int, list[float]] = {}
  for row in csv.reader(io.StringIO(csv_text)):
    if len(row) < 5:
      continue
    date_token = row[0].strip()
    if len(date_token) != 6 or not date_token.isdigit():
      continue
    try:
      market_excess = float(row[1].strip())
      risk_free = float(row[4].strip())
    except ValueError:
      continue
    year = int(date_token[:4])
    monthly_total_return = (market_excess + risk_free) / 100.0
    monthly_returns_by_year.setdefault(year, []).append(monthly_total_return)

  annual_returns: dict[int, float] = {}
  for year, monthly_returns in monthly_returns_by_year.items():
    annual_factor = 1.0
    for monthly_return in monthly_returns:
      annual_factor *= 1.0 + monthly_return
    annual_returns[year] = annual_factor - 1.0

  if not annual_returns:
    raise ValueError("Could not parse total-market historical returns table.")
  return annual_returns


def _load_us_inflation_by_year() -> dict[int, float]:
  csv_text = _fetch_text(US_CPI_URL)
  annual_cpi_sum: dict[int, float] = {}
  annual_cpi_count: dict[int, int] = {}
  for row in csv.DictReader(io.StringIO(csv_text)):
    value_text = row.get("CPIAUCSL", "")
    if value_text in ("", "."):
      continue
    date_text = row.get("observation_date", "")
    try:
      year = datetime.strptime(date_text, "%Y-%m-%d").year
      value = float(value_text)
    except ValueError:
      continue
    annual_cpi_sum[year] = annual_cpi_sum.get(year, 0.0) + value
    annual_cpi_count[year] = annual_cpi_count.get(year, 0) + 1

  annual_cpi_avg = {
    year: annual_cpi_sum[year] / annual_cpi_count[year]
    for year in annual_cpi_sum
    if annual_cpi_count[year] > 0
  }
  years = sorted(annual_cpi_avg)
  inflation_by_year: dict[int, float] = {}
  for index in range(1, len(years)):
    previous_year = years[index - 1]
    current_year = years[index]
    previous_cpi = annual_cpi_avg[previous_year]
    current_cpi = annual_cpi_avg[current_year]
    inflation_by_year[current_year] = current_cpi / previous_cpi - 1.0

  if not inflation_by_year:
    raise ValueError("Could not parse US inflation historical table.")
  return inflation_by_year


def load_historical_bootstrap_pairs(
  market_series: str,
  start_year: int | None = None,
  end_year: int | None = None,
) -> tuple[list[tuple[float, float]], tuple[int, int]]:
  if market_series == "sp500":
    market_by_year = _load_sp500_total_returns_by_year()
  elif market_series == "total-market":
    market_by_year = _load_total_market_returns_by_year()
  else:
    raise ValueError(f"Unsupported market_series: {market_series}")

  inflation_by_year = _load_us_inflation_by_year()
  aligned_years = sorted(
    year
    for year in market_by_year.keys() & inflation_by_year.keys()
    if (start_year is None or year >= start_year)
    and (end_year is None or year <= end_year)
  )
  if len(aligned_years) < 2:
    raise ValueError(
      "Not enough overlapping historical years between market and inflation series."
    )

  historical_pairs = [
    (market_by_year[year], inflation_by_year[year]) for year in aligned_years
  ]
  return historical_pairs, (aligned_years[0], aligned_years[-1])


def run_fire_simulations(
  initial_money: float,
  withdrawal_amount: float,
  inflation_mu: float,
  inflation_sigma: float,
  market_mu: float,
  market_sigma: float,
  simulations: int = 1000,
  max_years: int = 500,
  survival_probabilities: tuple[float, ...] = tuple(DEFAULT_SURVIVAL_PROBABILITIES),
  seed: int | None = None,
) -> SimulationResult:
  effective_max_years = min(max_years, HARD_MAX_YEARS)
  _validate_common_inputs(
    initial_money=initial_money,
    withdrawal_amount=withdrawal_amount,
    simulations=simulations,
    max_years=effective_max_years,
  )

  rng = random.Random(seed)
  years_to_zero = [
    _simulate_one_path(
      initial_money=initial_money,
      withdrawal_amount=withdrawal_amount,
      inflation_mu=inflation_mu,
      inflation_sigma=inflation_sigma,
      market_mu=market_mu,
      market_sigma=market_sigma,
      rng=rng,
      max_years=effective_max_years,
    )
    for _ in range(simulations)
  ]

  return _build_simulation_result(
    years_to_zero=years_to_zero,
    simulations=simulations,
    max_years=effective_max_years,
    survival_probabilities=survival_probabilities,
  )


def _simulate_one_path_block_bootstrap(
  initial_money: float,
  withdrawal_amount: float,
  historical_pairs: list[tuple[float, float]],
  block_years: int,
  rng: random.Random,
  max_years: int,
) -> int:
  balance = initial_money
  annual_withdrawal = withdrawal_amount
  years_elapsed = 0
  max_start_index = len(historical_pairs) - block_years

  while years_elapsed < max_years:
    block_start_index = rng.randint(0, max_start_index)
    block = historical_pairs[block_start_index : block_start_index + block_years]
    for market_return, inflation in block:
      years_elapsed += 1
      balance *= 1.0 + _clamp_rate(market_return)
      balance -= annual_withdrawal
      if balance <= 0.0:
        return years_elapsed
      annual_withdrawal *= 1.0 + _clamp_rate(inflation)
      if years_elapsed >= max_years:
        return max_years

  return max_years


def run_fire_simulations_block_bootstrap(
  initial_money: float,
  withdrawal_amount: float,
  historical_pairs: list[tuple[float, float]],
  block_years: int = 5,
  simulations: int = 1000,
  max_years: int = 500,
  survival_probabilities: tuple[float, ...] = tuple(DEFAULT_SURVIVAL_PROBABILITIES),
  seed: int | None = None,
) -> SimulationResult:
  effective_max_years = min(max_years, HARD_MAX_YEARS)
  _validate_common_inputs(
    initial_money=initial_money,
    withdrawal_amount=withdrawal_amount,
    simulations=simulations,
    max_years=effective_max_years,
  )
  if block_years <= 0:
    raise ValueError("block_years must be > 0")
  if len(historical_pairs) < block_years:
    raise ValueError("Not enough historical pairs for the requested block_years.")

  rng = random.Random(seed)
  years_to_zero = [
    _simulate_one_path_block_bootstrap(
      initial_money=initial_money,
      withdrawal_amount=withdrawal_amount,
      historical_pairs=historical_pairs,
      block_years=block_years,
      rng=rng,
      max_years=effective_max_years,
    )
    for _ in range(simulations)
  ]
  return _build_simulation_result(
    years_to_zero=years_to_zero,
    simulations=simulations,
    max_years=effective_max_years,
    survival_probabilities=survival_probabilities,
  )


def _resolve_distribution_inputs(
  preset_name: str | None,
  inflation_mu: float | None,
  inflation_sigma: float | None,
  market_mu: float | None,
  market_sigma: float | None,
) -> tuple[float, float, float, float]:
  preset = PRESETS.get(preset_name or "", {})
  resolved_inflation_mu = (
    inflation_mu if inflation_mu is not None else preset.get("inflation_mu")
  )
  resolved_inflation_sigma = (
    inflation_sigma if inflation_sigma is not None else preset.get("inflation_sigma")
  )
  resolved_market_mu = market_mu if market_mu is not None else preset.get("market_mu")
  resolved_market_sigma = (
    market_sigma if market_sigma is not None else preset.get("market_sigma")
  )

  missing_flags = []
  if resolved_inflation_mu is None:
    missing_flags.append("--inflation-mu")
  if resolved_inflation_sigma is None:
    missing_flags.append("--inflation-sigma")
  if resolved_market_mu is None:
    missing_flags.append("--market-mu")
  if resolved_market_sigma is None:
    missing_flags.append("--market-sigma")
  if missing_flags:
    raise ValueError(
      "Missing required parameters: "
      + ", ".join(missing_flags)
      + ". Provide them explicitly or use --preset."
    )

  return (
    resolved_inflation_mu,
    resolved_inflation_sigma,
    resolved_market_mu,
    resolved_market_sigma,
  )


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="FIRE depletion Monte Carlo simulator")
  parser.add_argument(
    "--model",
    choices=("normal", "block-bootstrap"),
    default="normal",
    help="Simulation model to use.",
  )
  parser.add_argument(
    "--initial-money",
    type=parse_money,
    required=True,
    help="Current portfolio amount. Supports k/m suffixes (e.g. 1m, 750k).",
  )
  parser.add_argument(
    "--withdrawal-amount",
    type=parse_money,
    required=True,
    help="Annual withdrawal amount. Supports k/m suffixes (e.g. 80k).",
  )
  parser.add_argument(
    "--preset",
    choices=sorted(PRESETS.keys()),
    default=None,
    help="Normal model only: use built-in assumptions (sp500 or total-market).",
  )
  parser.add_argument("--inflation-mu", type=float, default=None)
  parser.add_argument("--inflation-sigma", type=float, default=None)
  parser.add_argument("--market-mu", type=float, default=None)
  parser.add_argument("--market-sigma", type=float, default=None)
  parser.add_argument(
    "--market-series",
    choices=("sp500", "total-market"),
    default="total-market",
    help="Block-bootstrap only: historical market series.",
  )
  parser.add_argument(
    "--block-years",
    type=int,
    default=5,
    help="Block-bootstrap only: consecutive years per sampled block.",
  )
  parser.add_argument(
    "--historical-start-year",
    type=int,
    default=None,
    help="Block-bootstrap only: optional lower year bound for historical data.",
  )
  parser.add_argument(
    "--historical-end-year",
    type=int,
    default=None,
    help="Block-bootstrap only: optional upper year bound for historical data.",
  )
  parser.add_argument("--simulations", type=int, default=1000)
  parser.add_argument(
    "--max-years",
    type=int,
    default=HARD_MAX_YEARS,
    help=f"Simulation horizon in years (capped at {HARD_MAX_YEARS}).",
  )
  parser.add_argument(
    "--survival-probabilities",
    type=float,
    nargs="+",
    default=DEFAULT_SURVIVAL_PROBABILITIES,
    help=(
      "Survival levels in [0,1]. "
      "Default is 0.95 down to 0.0 by 0.05."
    ),
  )
  parser.add_argument("--seed", type=int, default=None)
  return parser


def main() -> None:
  parser = _build_parser()
  args = parser.parse_args()
  if (
    args.historical_start_year is not None
    and args.historical_end_year is not None
    and args.historical_start_year > args.historical_end_year
  ):
    parser.error("--historical-start-year must be <= --historical-end-year.")
  if args.max_years > HARD_MAX_YEARS:
    print(
      f"Note: requested max-years ({args.max_years}) is capped to "
      f"{HARD_MAX_YEARS}."
    )

  survival_probabilities = tuple(args.survival_probabilities)
  max_years = min(args.max_years, HARD_MAX_YEARS)
  model_details: str
  if args.model == "normal":
    try:
      inflation_mu, inflation_sigma, market_mu, market_sigma = (
        _resolve_distribution_inputs(
          preset_name=args.preset,
          inflation_mu=args.inflation_mu,
          inflation_sigma=args.inflation_sigma,
          market_mu=args.market_mu,
          market_sigma=args.market_sigma,
        )
      )
    except ValueError as error:
      parser.error(str(error))
    result = run_fire_simulations(
      initial_money=args.initial_money,
      withdrawal_amount=args.withdrawal_amount,
      inflation_mu=inflation_mu,
      inflation_sigma=inflation_sigma,
      market_mu=market_mu,
      market_sigma=market_sigma,
      simulations=args.simulations,
      max_years=max_years,
      survival_probabilities=survival_probabilities,
      seed=args.seed,
    )
    if args.preset is None:
      model_details = "normal (manual mu/sigma inputs)"
    else:
      model_details = f"normal (preset={args.preset})"
  else:
    try:
      historical_pairs, historical_year_range = load_historical_bootstrap_pairs(
        market_series=args.market_series,
        start_year=args.historical_start_year,
        end_year=args.historical_end_year,
      )
      result = run_fire_simulations_block_bootstrap(
        initial_money=args.initial_money,
        withdrawal_amount=args.withdrawal_amount,
        historical_pairs=historical_pairs,
        block_years=args.block_years,
        simulations=args.simulations,
        max_years=max_years,
        survival_probabilities=survival_probabilities,
        seed=args.seed,
      )
    except ValueError as error:
      parser.error(str(error))
    model_details = (
      "block-bootstrap "
      f"(market={args.market_series}, block-years={args.block_years}, "
      f"history={historical_year_range[0]}-{historical_year_range[1]})"
    )

  print(f"Model: {model_details}")
  print(f"Mean years until zero cash: {result.mean_years_to_zero:.2f}")
  print(f"Std years until zero cash: {result.std_years_to_zero:.2f}")
  for survival_probability, years in result.survival_horizons.items():
    print(
      f"Years with {survival_probability * 100:g}% survival: {years}"
    )
    if years == result.max_years:
      break
  _print_percentage_summary(result)
  plot_path = _write_survival_scatter_plot(
    years_to_zero=result.years_to_zero,
    output_path=DEFAULT_PLOT_OUTPUT,
    max_year=DEFAULT_PLOT_MAX_YEAR,
    points=DEFAULT_PLOT_POINTS,
  )
  print(f"Saved survival scatter plot: {plot_path}")


if __name__ == "__main__":
  main()
