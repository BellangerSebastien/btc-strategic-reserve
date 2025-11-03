"""
BTC Strategic Reserve — PyPortfolioOpt Implementation
-----------------------------------------------------
- Universe: MMF, ULTRA_SHORT, IG_SHORT, IBIT (Bitcoin ETF)
- Uses IBIT (iShares Bitcoin Trust) for institutional Bitcoin exposure
- Mean-CVaR optimization with PyPortfolioOpt's EfficientCVaR
- Evaluation by BTC tiers: {0.0, 0.5%, 1%, 1.5%, 2%, 2.5%, 3%}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import os

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientCVaR, expected_returns
from pypfopt.exceptions import OptimizationError
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

POOL_EUR = 1_000_000_000
NTL_EUR = 2_000_000_000
MONTHLY_OUTFLOWS_EUR = 500_000_000
MIN_COVER_MONTHS = 3

@dataclass
class Config:
    asset_order: Tuple[str, ...] = ("MMF", "ULTRA_SHORT", "IG_SHORT", "BTC_ETF")
    treasury_bounds: Dict[str, Tuple[float, float]] = None
    beta: float = 0.99
    risk_priority: float = 0.75
    btc_tiers: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
    weight_bounds: Tuple[float, float] = (0, 1)
    min_days: int = 250
    weight_cutoff: float = 1e-4
    verbose: bool = False
    risk_config: Optional[RiskEstimationConfig] = None

    def __post_init__(self):
        if self.treasury_bounds is None:
            self.treasury_bounds = {
                "MMF": (0.40, 0.55),
                "ULTRA_SHORT": (0.20, 0.30),
                "IG_SHORT": (0.20, 0.30)
            }
        if self.risk_config is None:
            self.risk_config = RiskEstimationConfig()


@dataclass
class RiskEstimationConfig:
    """
    Configuration for twelve-month risk estimation.

    Attributes:
        confidence_level: Confidence level for CVaR (e.g., 0.99 for 99%)
        method: Calculation method ("block_bootstrap" or "extreme_value_theory")
        number_of_paths: Number of simulated paths for bootstrap
        block_length_min: Minimum block length for bootstrap
        block_length_max: Maximum block length for bootstrap
        random_seed: Random seed for reproducibility
        threshold_quantile: Threshold quantile for extreme value theory
    """
    confidence_level: float = 0.99
    method: str = "block_bootstrap"
    number_of_paths: int = 50000
    block_length_min: int = 3
    block_length_max: int = 6
    random_seed: int = 42
    threshold_quantile: float = 0.97



def compute_conditional_value_at_risk_12m_block_bootstrap(
    portfolio_monthly_returns: pd.Series,
    confidence_level: float = 0.99,
    number_of_paths: int = 50000,
    block_length_min: int = 3,
    block_length_max: int = 6,
    random_seed: int = 42,
    aggregation: str = "compound"
) -> dict:
    """
    Computes the Value at Risk and Conditional Value at Risk over twelve months
    using block bootstrap, then composition of monthly returns.

    The method uses consecutive blocks of random length to preserve
    autocorrelation and regime dependencies in returns.

    Steps:
    1. Conversion to logarithmic returns
    2. Sampling of consecutive blocks of random length [min, max]
    3. Concatenation of blocks until obtaining 12 months per path
    4. Composition: return_12m = exp(sum(log_returns_12m)) - 1
    5. Loss calculation: loss = -return
    6. VaR = quantile at (1 - confidence_level)
    7. CVaR = average of losses beyond VaR

    Args:
        portfolio_monthly_returns: Series of monthly portfolio returns
        confidence_level: Confidence level (e.g., 0.99 for 99%)
        number_of_paths: Number of simulated paths
        block_length_min: Minimum block length
        block_length_max: Maximum block length
        random_seed: Random seed for reproducibility
        aggregation: Method for aggregating monthly returns over 12 months:
            - "compound": logarithmic composition exp(Σlog(1+r)) - 1 (default, rigorous)
            - "additive": additive sum Σr (Gaussian approximation for theoretical validation)

    Returns:
        dict containing:
            - "value_at_risk_12m": VaR over 12 months (positive = loss)
            - "conditional_value_at_risk_12m": CVaR over 12 months (positive = loss)
    """
    if aggregation not in ["compound", "additive"]:
        raise ValueError(f"aggregation must be 'compound' or 'additive', received: {aggregation}")

    np.random.seed(random_seed)

    log_returns = np.log(1 + portfolio_monthly_returns.values)
    n_observations = len(log_returns)

    if n_observations < 12:
        raise ValueError(f"Insufficient history: {n_observations} months (minimum: 12)")

    losses_12m = np.zeros(number_of_paths)

    for i in range(number_of_paths):
        log_returns_path = []

        while len(log_returns_path) < 12:
            block_length = np.random.randint(block_length_min, block_length_max + 1)
            start_idx = np.random.randint(0, n_observations - block_length + 1)
            block = log_returns[start_idx:start_idx + block_length]
            log_returns_path.extend(block)

        log_returns_12m = log_returns_path[:12]

        if aggregation == "compound":
            return_12m = np.expm1(np.sum(log_returns_12m))
        elif aggregation == "additive":
            simple_returns_12m = np.expm1(log_returns_12m)
            return_12m = np.sum(simple_returns_12m)
        else:
            raise ValueError(f"aggregation must be 'compound' or 'additive', received: {aggregation}")

        loss_12m = -return_12m
        losses_12m[i] = loss_12m

    var_threshold = np.percentile(losses_12m, confidence_level * 100)

    tail_losses = losses_12m[losses_12m >= var_threshold]
    cvar_12m = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold

    return {
        "value_at_risk_12m": var_threshold,
        "conditional_value_at_risk_12m": cvar_12m
    }


def compute_conditional_value_at_risk_12m_extreme_values(
    portfolio_monthly_returns: pd.Series,
    confidence_level: float = 0.99,
    threshold_quantile: float = 0.97,
    number_of_paths: int = 50000,
    random_seed: int = 42
) -> dict:
    """
    Computes VaR and CVaR over twelve months using Extreme Value Theory (EVT).

    Peaks Over Threshold approach:
    1. Calculate monthly losses
    2. Set threshold u = quantile(threshold_quantile) of losses
    3. Fit a Generalized Pareto distribution on exceedances
    4. Simulate monthly losses by mixing "body + tail"
    5. Generate 12-month paths and compose returns
    6. Calculate VaR and CVaR on the 12-month loss distribution

    Args:
        portfolio_monthly_returns: Series of monthly portfolio returns
        confidence_level: Confidence level (e.g., 0.99 for 99%)
        threshold_quantile: Quantile to define the exceedance threshold
        number_of_paths: Number of simulated paths
        random_seed: Random seed for reproducibility

    Returns:
        dict containing:
            - "value_at_risk_12m": VaR over 12 months (positive = loss)
            - "conditional_value_at_risk_12m": CVaR over 12 months (positive = loss)
    """
    np.random.seed(random_seed)

    monthly_losses = -portfolio_monthly_returns.values

    threshold = np.percentile(monthly_losses, threshold_quantile * 100)

    exceedances = monthly_losses[monthly_losses > threshold] - threshold

    if len(exceedances) < 10:
        print(f"WARNING: Warning: only {len(exceedances)} exceedances for EVT")
        return compute_conditional_value_at_risk_12m_block_bootstrap(
            portfolio_monthly_returns, confidence_level, number_of_paths,
            block_length_min=3, block_length_max=6, random_seed=random_seed
        )

    try:
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    except Exception as e:
        print(f"WARNING: GPD fitting failed: {e}. Switching to bootstrap.")
        return compute_conditional_value_at_risk_12m_block_bootstrap(
            portfolio_monthly_returns, confidence_level, number_of_paths,
            block_length_min=3, block_length_max=6, random_seed=random_seed
        )

    prob_exceedance = len(exceedances) / len(monthly_losses)

    body_losses = monthly_losses[monthly_losses <= threshold]

    losses_12m = np.zeros(number_of_paths)

    for i in range(number_of_paths):
        log_returns_path = []

        for _ in range(12):
            if np.random.rand() < prob_exceedance:
                exceedance = stats.genpareto.rvs(shape, loc=loc, scale=scale)
                loss_monthly = threshold + exceedance
            else:
                loss_monthly = np.random.choice(body_losses)

            return_monthly = -loss_monthly
            log_return = np.log(1 + return_monthly)
            log_returns_path.append(log_return)

        return_12m = np.exp(np.sum(log_returns_path)) - 1
        loss_12m = -return_12m
        losses_12m[i] = loss_12m

    var_threshold = np.percentile(losses_12m, confidence_level * 100)

    tail_losses = losses_12m[losses_12m >= var_threshold]
    cvar_12m = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold

    return {
        "value_at_risk_12m": var_threshold,
        "conditional_value_at_risk_12m": cvar_12m
    }



def fetch_prices_eur(start: str = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches daily prices for the portfolio with complete treasury structure.

    Returns a DataFrame with columns:
    - MMF: Liquid core (Money Market Fund)
    - ULTRA_SHORT: Enhanced cash (ultra-short bonds)
    - IG_SHORT: Sovereign/short-term IG bonds
    - BTC_ETF: IBIT (iShares Bitcoin Trust) - institutional Bitcoin ETF

    Note: IBIT trades only on business days, eliminating the 24/7 calendar issue
    """
    tickers = {
        "CSH.PA": "MMF",
        "IS3M.DE": "ULTRA_SHORT",
        "SE15.L": "IG_SHORT",
        "BTCE.DE": "BTC_ETF"
    }

    data = yf.download(
        list(tickers.keys()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise RuntimeError("No data retrieved from Yahoo Finance")

    if len(tickers) == 1:
        prices = pd.DataFrame(data["Adj Close"] if "Adj Close" in data.columns else data["Close"])
        prices.columns = [list(tickers.values())[0]]
    else:
        prices = (data["Adj Close"] if "Adj Close" in data.columns else data["Close"])
        prices = prices.rename(columns=tickers)

    prices = prices.ffill().dropna()

    if prices.empty:
        raise RuntimeError("No valid data after processing")

    expected_cols = ["MMF", "ULTRA_SHORT", "IG_SHORT", "BTC_ETF"]
    available_cols = [col for col in expected_cols if col in prices.columns]

    if len(available_cols) < len(expected_cols):
        print(f"WARNING: Missing columns. Available: {available_cols}")
        print(f"    Expected: {expected_cols}")

    prices = prices[available_cols].dropna()

    return prices



def optimize_cvar_pypfopt(
    prices: pd.DataFrame,
    cfg: Config,
    wbtc_fixed: Optional[float] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Optimizes the portfolio using PyPortfolioOpt's EfficientCVaR.

    Returns:
        weights: Dict of optimal weights
        metrics: Dict of performance metrics
    """
    monthly_prices = prices.resample('ME').last()
    mu = expected_returns.mean_historical_return(monthly_prices, frequency=12)
    returns = expected_returns.returns_from_prices(monthly_prices)

    ef_cvar = EfficientCVaR(
        mu,
        returns,
        beta=cfg.beta,
        weight_bounds=cfg.weight_bounds,
        verbose=cfg.verbose
    )

    if wbtc_fixed is not None:
        if "BTC_ETF" in prices.columns:
            btc_idx = list(prices.columns).index("BTC_ETF")
            ef_cvar.add_constraint(lambda w: w[btc_idx] == wbtc_fixed)
            remaining_weight = 1.0 - wbtc_fixed
            for asset, (min_bound, max_bound) in cfg.treasury_bounds.items():
                if asset in prices.columns:
                    asset_idx = list(prices.columns).index(asset)
                    ef_cvar.add_constraint(lambda w, idx=asset_idx, min_b=min_bound:
                                          w[idx] >= min_b * remaining_weight)
                    ef_cvar.add_constraint(lambda w, idx=asset_idx, max_b=max_bound:
                                          w[idx] <= max_b * remaining_weight)

    try:
        if cfg.risk_priority > 0.5:
            weights = ef_cvar.min_cvar()
        else:
            target_return = mu.mean() * (1 - cfg.risk_priority)
            weights = ef_cvar.efficient_return(target_return)

        clean_weights = ef_cvar.clean_weights(cutoff=cfg.weight_cutoff)

        ret, cvar_monthly = ef_cvar.portfolio_performance(verbose=False)

        weights_array = np.array([clean_weights[asset] for asset in prices.columns])
        portfolio_returns_array = returns @ weights_array
        portfolio_returns = pd.Series(portfolio_returns_array, index=returns.index)

        var_monthly = np.percentile(portfolio_returns.values, (1 - cfg.beta) * 100)

        if cfg.risk_config.method == "block_bootstrap":
            risk_12m = compute_conditional_value_at_risk_12m_block_bootstrap(
                portfolio_returns,
                confidence_level=cfg.risk_config.confidence_level,
                number_of_paths=cfg.risk_config.number_of_paths,
                block_length_min=cfg.risk_config.block_length_min,
                block_length_max=cfg.risk_config.block_length_max,
                random_seed=cfg.risk_config.random_seed
            )
        elif cfg.risk_config.method == "extreme_value_theory":
            risk_12m = compute_conditional_value_at_risk_12m_extreme_values(
                portfolio_returns,
                confidence_level=cfg.risk_config.confidence_level,
                threshold_quantile=cfg.risk_config.threshold_quantile,
                number_of_paths=cfg.risk_config.number_of_paths,
                random_seed=cfg.risk_config.random_seed
            )
        else:
            risk_12m = compute_conditional_value_at_risk_12m_block_bootstrap(
                portfolio_returns,
                confidence_level=cfg.risk_config.confidence_level,
                number_of_paths=cfg.risk_config.number_of_paths,
                block_length_min=cfg.risk_config.block_length_min,
                block_length_max=cfg.risk_config.block_length_max,
                random_seed=cfg.risk_config.random_seed
            )

        cvar_12m = risk_12m["conditional_value_at_risk_12m"]
        var_12m = risk_12m["value_at_risk_12m"]

        sharpe_cvar_12m = ret / cvar_12m if cvar_12m > 0 else 0

        metrics = {
            "expected_return": ret,
            "cvar_monthly": cvar_monthly,
            "var_monthly": -var_monthly,
            "cvar_12m": cvar_12m,
            "var_12m": var_12m,
            "sharpe_cvar_12m": sharpe_cvar_12m,
            "cvar": cvar_12m,
            "var": var_12m,
            "sharpe_cvar": sharpe_cvar_12m,
            "status": "optimal"
        }

        return clean_weights, metrics

    except (OptimizationError, Exception) as e:
        default_weights = {asset: 0.0 for asset in prices.columns}
        default_weights["CASH"] = 1.0

        metrics = {
            "expected_return": 0.0,
            "cvar": 0.0,
            "cvar_monthly": 0.0,
            "var": 0.0,
            "sharpe_cvar": 0.0,
            "status": f"failed: {str(e)}"
        }

        return default_weights, metrics


def volatility_risk_contributions(
    prices: pd.DataFrame,
    weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculates the contribution to volatility of each asset in the portfolio.

    Exact mathematical formula:
    - Absolute contribution to σ: RC_i(σ) = w_i × (Σw)_i / σ_p
    - Share in % of total risk: RC_i(%) = w_i × (Σw)_i / σ_p²

    where (Σw)_i is the i-th element of the vector Σw (matrix-vector product).
    Contributions sum to σ_p and shares in % sum to 100%.

    WARNING: IMPORTANT: This function calculates contributions to VOLATILITY (σ).
    For full consistency with Mean-CVaR optimization, use instead
    the `cvar_risk_contributions` function which calculates Euler contributions to CVaR.

    Args:
        prices: DataFrame with historical asset prices
        weights: Dict with the weight of each asset

    Returns:
        Dict with the volatility contribution in % for each asset
    """
    monthly_prices = prices.resample('ME').last()
    returns = monthly_prices.pct_change().dropna()

    assets = list(prices.columns)
    w = np.array([weights.get(asset, 0.0) for asset in assets])

    cov_matrix = returns.cov() * 12

    portfolio_returns = returns @ w

    portfolio_variance = portfolio_returns.var() * 12
    portfolio_std = np.sqrt(portfolio_variance)

    risk_contributions = {}

    for i, asset in enumerate(assets):
        cov_i_p = returns[asset].cov(portfolio_returns) * 12
        rc_pct = w[i] * cov_i_p / portfolio_variance if portfolio_variance > 0 else 0
        risk_contributions[asset] = rc_pct * 100  # In percentage
    return risk_contributions


def cvar_risk_contributions(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    beta: float = 0.99
) -> Dict[str, float]:
    """
    Calculates Euler contributions to CVaR_β using the Acerbi-Tasche method.

    This method is consistent with Mean-CVaR optimization and decomposes CVaR
    of the portfolio into additive contributions of each asset.

    Euler formula for CVaR:
    - CVaR_p = Σ_i (w_i × E[-r_i | r_p ≤ VaR_β])
    - Asset i contribution: RC_i = w_i × E[-r_i | r_p ≤ VaR_β]
    - Share in %: RC_i% = RC_i / CVaR_p × 100

    Contributions sum to 100% of total CVaR.
    Convention: loss = -return, so CVaR > 0 represents a loss.

    Args:
        prices: DataFrame with historical asset prices
        weights: Dict with the weight of each asset
        beta: Confidence level for CVaR (default: 0.99 for 99%)

    Returns:
        Dict with CVaR contribution in % for each asset
    """
    monthly = prices.resample('ME').last().pct_change().dropna()
    assets = list(monthly.columns)
    w = np.array([weights.get(a, 0.0) for a in assets])

    p = (monthly * w).sum(axis=1)

    var_thresh = p.quantile(1 - beta)
    tail = p <= var_thresh

    cvar_p = -(p[tail]).mean()

    if cvar_p <= 0 or tail.sum() == 0:
        return {a: 0.0 for a in assets}

    contrib = -w * monthly[tail].mean().values

    share = contrib / cvar_p

    return {a: 100.0 * s for a, s in zip(assets, share)}



def evaluate_btc_grid_pypfopt(
    prices: pd.DataFrame,
    cfg: Config
) -> pd.DataFrame:
    """
    Evaluates different BTC allocation levels using PyPortfolioOpt.
    """
    results = []

    for wbtc in cfg.btc_tiers:
        weights, metrics = optimize_cvar_pypfopt(prices, cfg, wbtc_fixed=wbtc)

        ntl_check = ntl_at_risk_check(
            cvar_monthly=metrics.get("cvar_monthly", 0),  # Use monthly CVaR
            pool_eur=POOL_EUR,
            ntl_eur=NTL_EUR,
            monthly_outflows_eur=MONTHLY_OUTFLOWS_EUR,
            min_cover_months=MIN_COVER_MONTHS
        )

        vol_risk_contributions = volatility_risk_contributions(prices, weights)
        cvar_risk_contribs = cvar_risk_contributions(prices, weights, beta=cfg.beta)

        row = {
            "wBTC": wbtc,
            "expected_return": metrics["expected_return"],
            "CVaR_monthly": metrics["cvar_monthly"],
            "VaR_monthly": metrics["var_monthly"],
            "CVaR_12m": metrics["cvar_12m"],
            "VaR_12m": metrics["var_12m"],
            "Sharpe_CVaR_12m": metrics["sharpe_cvar_12m"],
            "CVaR": metrics["cvar"],
            "VaR": metrics["var"],
            "Sharpe_CVaR": metrics["sharpe_cvar"],
            "NTL_loss_EUR": ntl_check["loss_eur"],
            "NTL_post_EUR": ntl_check["ntl_post"],
            "cover_months": ntl_check["cover_months"],
            "NTL_ok": ntl_check["ok"],
            **{f"RC_vol_{asset}": vol_risk_contributions[asset] for asset in prices.columns},
            **{f"RC_cvar_{asset}": cvar_risk_contribs[asset] for asset in prices.columns},
            **{f"w_{asset}": weights[asset] for asset in prices.columns},
            "status": metrics["status"]
        }
        results.append(row)

    df = pd.DataFrame(results).set_index("wBTC")
    return df


def ntl_at_risk_check(cvar_monthly: float,
                      pool_eur: float,
                      ntl_eur: float,
                      monthly_outflows_eur: float,
                      min_cover_months: int = 3):
    """
    cvar_monthly : Portfolio CVaR over 1 month (positive, e.g. 0.08 = -8% in stress month)
    Returns : dict with loss_eur, ntl_post, cover_months, ok (bool)
    """
    loss_eur = cvar_monthly * pool_eur
    ntl_post = max(ntl_eur - loss_eur, 0.0)
    cover_months = ntl_post / monthly_outflows_eur if monthly_outflows_eur > 0 else float('inf')
    ok = cover_months >= min_cover_months
    return {
        "loss_eur": loss_eur,
        "ntl_post": ntl_post,
        "cover_months": cover_months,
        "ok": ok
    }


def run_pypfopt_pipeline(
    start: str = "2020-01-01",
    end: Optional[str] = None,
    cfg: Optional[Config] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the complete pipeline with PyPortfolioOpt.

    Returns:
        Tuple of (df_results, prices)
    """
    if cfg is None:
        cfg = Config()

    print("Fetching market data...")
    prices = fetch_prices_eur(start=start, end=end)

    if len(prices) < cfg.min_days:
        raise ValueError(f"Insufficient history: {len(prices)} days (min: {cfg.min_days})")

    print(f"[PASS] {len(prices)} days of data retrieved")
    print(f"Assets: {list(prices.columns)}")

    print("\nOptimization in progress...")
    df_results = evaluate_btc_grid_pypfopt(prices, cfg)

    return df_results, prices



def display_results_pypfopt(df: pd.DataFrame, cfg: Config):
    """
    Displays results with enhanced formatting.
    """
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # No width limit
    pd.set_option('display.max_colwidth', None) # No column width limit
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1 else f'{x:.2f}')

    print("\n" + "="*80)
    print(" "*15 + "BTC STRATEGIC RESERVE - PYPFOPT ANALYSIS")
    print(" "*20 + f"Confidence Level: {cfg.beta:.0%}")
    print(" "*15 + f"Risk Method: {cfg.risk_config.method}")
    print("="*80)

    print("\nOPTIMIZATION RESULTS (PyPortfolioOpt)")
    print("-"*80)

    display_df = df.copy()

    for col in display_df.columns:
        if col.startswith('w_'):
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        elif col in ['expected_return', 'CVaR', 'VaR', 'CVaR_monthly', 'VaR_monthly', 'CVaR_12m', 'VaR_12m']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        elif col in ['Sharpe_CVaR', 'Sharpe_CVaR_12m']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        elif col == 'NTL_loss_EUR':
            display_df[col] = display_df[col].apply(lambda x: f"{x/1_000_000:.1f}M€")
        elif col == 'NTL_post_EUR':
            display_df[col] = display_df[col].apply(lambda x: f"{x/1_000_000:.0f}M€")
        elif col == 'cover_months':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if x != float('inf') else "∞")
        elif col == 'NTL_ok':
            display_df[col] = display_df[col].apply(lambda x: "[OK]" if x else "[X]")

    display_df.index = display_df.index.map(lambda x: f"{x:.2%}")

    weight_cols = [col for col in display_df.columns if col.startswith('w_')]
    display_cols = [
        'expected_return',
        'CVaR_monthly', 'VaR_monthly',  # Monthly metrics
        'CVaR_12m', 'VaR_12m', 'Sharpe_CVaR_12m',  # 12-month compounded metrics
        'NTL_loss_EUR', 'cover_months', 'NTL_ok'
    ] + weight_cols
    available_display_cols = [col for col in display_cols if col in display_df.columns]

    print(display_df[available_display_cols].to_string())
    print("\nNote: CVaR_monthly/VaR_monthly = monthly risk | CVaR_12m/VaR_12m = 12-month compounded risk (no √12)")

    valid_data = df[df['status'] == 'optimal']

    if not valid_data.empty:
        print("\n" + "="*80)
        print(" "*20 + "TREASURY LIQUIDITY ANALYSIS (NTL-at-Risk)")
        print("="*80)

        feasible_data = valid_data[valid_data['NTL_ok'] == True]

        print(f"\nLIQUIDITY CONSTRAINTS")
        print(f"   • Portfolio size: {POOL_EUR/1_000_000:.0f}M€")
        print(f"   • Current NTL: {NTL_EUR/1_000_000:.0f}M€")
        print(f"   • Monthly outflows: {MONTHLY_OUTFLOWS_EUR/1_000_000:.0f}M€")
        print(f"   • Required coverage: ≥ {MIN_COVER_MONTHS} months")

        print(f"\nFEASIBILITY CHECK")
        print(f"   • Total allocations tested: {len(valid_data)}")
        print(f"   • Allocations respecting NTL constraint: {len(feasible_data)}")
        print(f"   • Allocations violating constraint: {len(valid_data) - len(feasible_data)}")

        print("\n" + "="*80)
        print(" "*25 + "OPTIMAL ANALYSIS")
        print("="*80)

        if not feasible_data.empty:
            idx_optimal = feasible_data['Sharpe_CVaR'].idxmax()
            optimal_row = feasible_data.loc[idx_optimal]
            optimization_status = "[PASS] WITH NTL CONSTRAINT"
        else:
            idx_optimal = valid_data['cover_months'].idxmax()
            optimal_row = valid_data.loc[idx_optimal]
            optimization_status = "WARNING: NO FEASIBLE SOLUTION - SHOWING BEST COVERAGE"

        print(f"\nOPTIMAL ALLOCATION {optimization_status}: {idx_optimal:.2%} BTC")
        print(f"   • Expected annualized return: {optimal_row['expected_return']:.2%}")
        print(f"   • CVaR ({cfg.beta:.0%}) monthly: {optimal_row['CVaR_monthly']:.2%}")
        print(f"   • CVaR ({cfg.beta:.0%}) 12-month compounded: {optimal_row['CVaR_12m']:.2%}")
        print(f"   • VaR ({cfg.beta:.0%}) 12-month compounded: {optimal_row['VaR_12m']:.2%}")
        print(f"   • Sharpe-CVaR ratio (12m): {optimal_row['Sharpe_CVaR_12m']:.3f}")
        print(f"\nNTL IMPACT")
        print(f"   • Monthly stress loss: {optimal_row['NTL_loss_EUR']/1_000_000:.1f}M€")
        print(f"   • NTL post-stress: {optimal_row['NTL_post_EUR']/1_000_000:.0f}M€")
        print(f"   • Coverage post-stress: {optimal_row['cover_months']:.1f} months")
        print(f"   • Constraint satisfied: {optimal_row['NTL_ok']}")

        if not feasible_data.empty:
            threshold_cvar = feasible_data['CVaR'].quantile(0.5)
            conservative_mask = feasible_data['CVaR'] <= threshold_cvar

            if conservative_mask.any():
                wbtc_min = feasible_data[conservative_mask].index.min()
                wbtc_conservative_max = feasible_data[conservative_mask].index.max()
            else:
                wbtc_min = feasible_data.index.min()
                wbtc_conservative_max = feasible_data.index.min()

            sharpe_threshold = optimal_row['Sharpe_CVaR'] * 0.8
            aggressive_mask = feasible_data['Sharpe_CVaR'] >= sharpe_threshold
            wbtc_max = feasible_data[aggressive_mask].index.max() if aggressive_mask.any() else feasible_data.index.max()

            print(f"\nRECOMMENDED RANGES (NTL-CONSTRAINED)")
            print(f"   • Conservative: {wbtc_min:.2%} - {wbtc_conservative_max:.2%}")
            print(f"   • Optimal: {max(wbtc_min, idx_optimal - 0.005):.2%} - {min(wbtc_max, idx_optimal + 0.005):.2%}")
            print(f"   • Aggressive: {idx_optimal:.2%} - {wbtc_max:.2%}")
        else:
            print(f"\nWARNING: NO FEASIBLE ALLOCATION RANGE")
            print(f"   All allocations violate the {MIN_COVER_MONTHS}-month coverage requirement")
            print(f"   Consider reducing portfolio size or increasing NTL buffer")


        print("\nRISK-RETURN PROFILE WITH NTL STATUS")
        print("-"*60)

        max_return = valid_data['expected_return'].max()
        for idx, row in valid_data.iterrows():
            wbtc_pct = idx * 100
            bar_length = int((row['expected_return'] / max_return) * 40) if max_return > 0 else 0

            ntl_indicator = "[OK]" if row['NTL_ok'] else "[X]"

            if not feasible_data.empty:
                threshold_cvar_chart = feasible_data['CVaR'].quantile(0.5)
                sharpe_threshold_chart = feasible_data['Sharpe_CVaR'].max() * 0.8
                risk_indicator = "[LOW]" if row['CVaR'] <= threshold_cvar_chart else "[MED]" if row['Sharpe_CVaR'] >= sharpe_threshold_chart else "[HIGH]"
            else:
                risk_indicator = "[HIGH]"

            optimal_marker = " ← OPTIMAL" if idx == idx_optimal else ""
            print(f"  {wbtc_pct:4.2f}% {ntl_indicator} {risk_indicator} {'█' * bar_length} {row['expected_return']:.3f}{optimal_marker}")

        print("\n" + "="*80)
        print("INSIGHTS (PyPortfolioOpt with NTL-at-Risk)")
        print("-"*80)

        if not feasible_data.empty:
            print(f"• Optimal allocation with NTL constraint: {idx_optimal:.2%} BTC")
            print(f"• {len(feasible_data)} allocations maintain ≥{MIN_COVER_MONTHS} months coverage after stress")
            print(f"• Maximum BTC allocation respecting liquidity: {feasible_data.index.max():.2%}")
        else:
            print(f"• WARNING: No allocation meets the {MIN_COVER_MONTHS}-month coverage requirement")
            print(f"• Best coverage achieved: {optimal_row['cover_months']:.1f} months at {idx_optimal:.2%} BTC")
            print(f"• Consider adjusting portfolio size or NTL buffer")

        print("\nLEGEND")
        print("  [PASS] Meets NTL coverage requirement")
        print("  [FAIL] Violates NTL coverage requirement")
        print("  [LOW] Low risk (CVaR < median of feasible)")
        print("  [MED] Moderate risk (Sharpe > 80% of best feasible)")
        print("  [HIGH] High risk or unfeasible")

    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.float_format')

    print("\n" + "="*80)



def ensure_output_directory(output_dir: str = "btc_reserve_plots") -> str:
    """
    Creates the output directory if it doesn't exist.
    Returns the absolute path to the directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def plot_efficient_frontier(valid_data: pd.DataFrame, feasible_data: pd.DataFrame,
                            idx_optimal: float, cfg: Config, output_path: str):
    """
    Graph 1: Mean-CVaR Efficient Frontier with BTC Allocation
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    btc_pct = valid_data.index * 100

    cvar_col = 'CVaR_12m' if 'CVaR_12m' in valid_data.columns else 'CVaR'

    scatter = ax.scatter(valid_data[cvar_col] * 100,
                        valid_data['expected_return'] * 100,
                        c=btc_pct, s=0.1, alpha=0,
                        cmap='viridis', edgecolors='none', zorder=2)

    ax.plot(valid_data[cvar_col] * 100, valid_data['expected_return'] * 100,
           '-', linewidth=3, color='#1f77b4', alpha=0.6, zorder=1, label='Efficient Frontier')

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('BTC Allocation (%)', fontsize=10, fontweight='bold')

    violations = valid_data[valid_data['NTL_ok'] == False]
    if not violations.empty:
        ax.scatter(violations[cvar_col] * 100, violations['expected_return'] * 100,
                  marker='x', s=100, c='red', linewidth=2, alpha=0.8, zorder=3,
                  label='[FAIL] NTL Violation')

    optimal_row = valid_data.loc[idx_optimal]
    ax.scatter(optimal_row[cvar_col] * 100, optimal_row['expected_return'] * 100,
              marker='*', s=300, c='gold', edgecolors='black', linewidth=2,
              label=f'* Optimal: {idx_optimal:.2%} BTC', zorder=5)

    ax.set_xlabel(f'CVaR {cfg.beta:.0%} (12-month compounded) (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean-CVaR Efficient Frontier with BTC Allocation',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_portfolio_composition(valid_data: pd.DataFrame, output_path: str):
    """
    Graph 2: Portfolio Composition (Stacked Area)
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    weight_cols = [col for col in valid_data.columns if col.startswith('w_')]
    asset_names = [col.replace('w_', '') for col in weight_cols]

    btc_allocations = valid_data.index * 100
    weights_array = valid_data[weight_cols].values * 100

    colors_assets = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(asset_names)]

    ax.stackplot(btc_allocations, weights_array.T,
                labels=asset_names, colors=colors_assets, alpha=0.8)

    ax.set_xlabel('BTC Allocation (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Portfolio Weight (%)', fontsize=11, fontweight='bold')
    ax.set_title('Portfolio Composition', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_sharpe_cvar(valid_data: pd.DataFrame, feasible_data: pd.DataFrame,
                     idx_optimal: float, output_path: str):
    """
    Graph 3: Sharpe-CVaR Ratio by BTC Tier
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    btc_allocations = valid_data.index * 100
    sharpe_values = valid_data['Sharpe_CVaR']
    colors_sharpe = ['gold' if idx == idx_optimal else '#1f77b4' for idx in valid_data.index]

    ax.bar(btc_allocations, sharpe_values, color=colors_sharpe, alpha=0.7,
           edgecolor='black', linewidth=1)

    if not feasible_data.empty:
        threshold = feasible_data['Sharpe_CVaR'].max() * 0.8
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2,
                  label=f'80% Optimal Threshold', alpha=0.7)

    ax.set_xlabel('BTC Allocation (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sharpe-CVaR Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_return_vs_btc(valid_data: pd.DataFrame, idx_optimal: float,
                       optimal_row: pd.Series, output_path: str):
    """
    Graph 4: Expected Return vs BTC Allocation
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    btc_allocations = valid_data.index * 100
    ax.plot(btc_allocations, valid_data['expected_return'] * 100,
           marker='o', linewidth=2, color='#2ca02c', markersize=5)
    ax.scatter(idx_optimal * 100, optimal_row['expected_return'] * 100,
              marker='*', s=300, c='gold', edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlabel('BTC Allocation (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=11, fontweight='bold')
    ax.set_title('Expected Return vs BTC Allocation', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_cvar_vs_btc(valid_data: pd.DataFrame, idx_optimal: float,
                     optimal_row: pd.Series, output_path: str):
    """
    Graph 5: CVaR vs BTC Allocation (12 mois composés)
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    cvar_col = 'CVaR_12m' if 'CVaR_12m' in valid_data.columns else 'CVaR'

    btc_allocations = valid_data.index * 100
    ax.plot(btc_allocations, valid_data[cvar_col] * 100,
           marker='o', linewidth=2, color='#d62728', markersize=5)
    ax.scatter(idx_optimal * 100, optimal_row[cvar_col] * 100,
              marker='*', s=300, c='gold', edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlabel('BTC Allocation (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('CVaR (12-month compounded) (%)', fontsize=11, fontweight='bold')
    ax.set_title('CVaR vs BTC Allocation', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_combined_metrics(valid_data: pd.DataFrame, idx_optimal: float, output_path: str):
    """
    Graph 6: Combined Metrics - Sharpe, Return, and NTL Loss
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax_right = ax.twinx()

    btc_allocations = valid_data.index * 100
    sharpe_values = valid_data['Sharpe_CVaR']

    line1 = ax.plot(btc_allocations, sharpe_values,
                   'b-', linewidth=2.5, label='Sharpe-CVaR Ratio', marker='o', markersize=4)
    line2 = ax.plot(btc_allocations, valid_data['expected_return'] * 100,
                   'g-', linewidth=2.5, label='Expected Return (%)', marker='s', markersize=4)

    line3 = ax_right.plot(btc_allocations, valid_data['NTL_loss_EUR'] / 1_000_000,
                         'r-', linewidth=2.5, label='NTL Loss (M EUR)', marker='^', markersize=4)

    ax.axvline(x=idx_optimal * 100, color='gold', linestyle='--', linewidth=2,
              alpha=0.7, label='Optimal Allocation')

    violations_idx = valid_data[valid_data['NTL_ok'] == False].index
    if len(violations_idx) > 0:
        violations_x = violations_idx * 100
        violations_y = valid_data.loc[violations_idx, 'NTL_loss_EUR'] / 1_000_000
        ax_right.scatter(violations_x, violations_y, marker='x', s=100, c='darkred',
                       linewidth=2, zorder=5, label='NTL Violation')

    ax.set_xlabel('BTC Allocation (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sharpe-CVaR Ratio / Return (%)', fontsize=11, fontweight='bold', color='black')
    ax_right.set_ylabel('NTL Loss (M EUR)', fontsize=11, fontweight='bold', color='red')
    ax.set_title('Combined Metrics: Performance vs Risk', fontsize=12, fontweight='bold', pad=10)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_right.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def calculate_drawdowns(prices: pd.Series) -> pd.Series:
    """
    Calculates the drawdown series for a price series.
    Drawdown = (price - running_max) / running_max
    """
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return drawdown


def plot_btc_drawdowns(prices: pd.DataFrame, output_path: str):
    """
    Graph 7: Bitcoin Drawdown Timeline
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    if 'BTC_ETF' not in prices.columns:
        print("  ⚠ BTC_ETF not found in prices, skipping drawdown chart")
        return

    btc_prices = prices['BTC_ETF']
    drawdowns = calculate_drawdowns(btc_prices)

    ax.fill_between(drawdowns.index, drawdowns * 100, 0,
                     where=(drawdowns < -0.3), color='darkred', alpha=0.3, label='Drawdown > 30%')
    ax.fill_between(drawdowns.index, drawdowns * 100, 0,
                     where=(drawdowns >= -0.3) & (drawdowns < -0.1),
                     color='orange', alpha=0.3, label='Drawdown 10-30%')
    ax.fill_between(drawdowns.index, drawdowns * 100, 0,
                     where=(drawdowns >= -0.1), color='lightgreen', alpha=0.3, label='Drawdown < 10%')

    ax.plot(drawdowns.index, drawdowns * 100, color='black', linewidth=1.5, label='BTC Drawdown')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-30, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7)

    max_dd = drawdowns.min() * 100
    max_dd_date = drawdowns.idxmin()
    ax.annotate(f'Max DD: {max_dd:.1f}%',
                xy=(max_dd_date, max_dd),
                xytext=(max_dd_date, max_dd - 10),
                fontsize=10, fontweight='bold', color='darkred',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax.set_title('Bitcoin Maximum Drawdown Over Time', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_return_distributions(prices: pd.DataFrame, output_path: str):
    """
    Graph 8: Monthly Returns Distribution Comparison
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()

    if 'BTC_ETF' in monthly_returns.columns:
        btc_returns = monthly_returns['BTC_ETF'] * 100
        axes[0].hist(btc_returns, bins=30, alpha=0.7, color='#d62728', edgecolor='black')
        axes[0].axvline(btc_returns.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {btc_returns.mean():.2f}%')
        axes[0].axvline(btc_returns.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {btc_returns.median():.2f}%')
        axes[0].set_xlabel('Monthly Return (%)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('BTC_ETF Monthly Returns Distribution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='y')

    traditional_assets = [col for col in monthly_returns.columns if col != 'BTC_ETF']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, asset in enumerate(traditional_assets):
        if i < len(colors):
            asset_returns = monthly_returns[asset] * 100
            axes[1].hist(asset_returns, bins=20, alpha=0.6, color=colors[i],
                        edgecolor='black', label=asset)

    axes[1].set_xlabel('Monthly Return (%)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Traditional Assets Monthly Returns', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_volatility_comparison(prices: pd.DataFrame, output_path: str):
    """
    Graph 9: Annualized Volatility Comparison
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    annualized_vols = monthly_returns.std() * np.sqrt(12) * 100

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(range(len(annualized_vols)), annualized_vols.values,
                   color=colors[:len(annualized_vols)], alpha=0.8, edgecolor='black', linewidth=1.5)

    for i, (asset, vol) in enumerate(annualized_vols.items()):
        ax.text(i, vol + 1, f'{vol:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(annualized_vols)))
    ax.set_xticklabels(annualized_vols.index, fontsize=11, fontweight='bold')
    ax.set_ylabel('Annualized Volatility (%)', fontsize=11, fontweight='bold')
    ax.set_title('Annualized Volatility Comparison (2020-Present)', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_worst_months(prices: pd.DataFrame, output_path: str):
    """
    Graph 10: Worst Monthly Return by Asset
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    worst_months = monthly_returns.min() * 100

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(range(len(worst_months)), worst_months.values,
                   color=colors[:len(worst_months)], alpha=0.8, edgecolor='black', linewidth=1.5)

    for i, (asset, ret) in enumerate(worst_months.items()):
        ax.text(i, ret - 2, f'{ret:.1f}%', ha='center', fontsize=10, fontweight='bold', color='white')

    ax.set_xticks(range(len(worst_months)))
    ax.set_xticklabels(worst_months.index, fontsize=11, fontweight='bold')
    ax.set_ylabel('Worst Monthly Return (%)', fontsize=11, fontweight='bold')
    ax.set_title('Worst Historical Month by Asset (2020-Present)', fontsize=12, fontweight='bold', pad=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_btc_price_with_risk_zones(prices: pd.DataFrame, output_path: str):
    """
    Graph 11: BTC Price with Volatility Risk Zones
    """
    if 'BTC_ETF' not in prices.columns:
        print("  ⚠ BTC_ETF not found in prices, skipping risk zones chart")
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    btc_prices = prices['BTC_ETF']
    btc_returns = btc_prices.pct_change()
    rolling_vol = btc_returns.rolling(window=30).std() * np.sqrt(252)

    normalized_price = (btc_prices / btc_prices.iloc[0]) * 100

    vol_low = rolling_vol.quantile(0.33)
    vol_high = rolling_vol.quantile(0.67)

    low_vol_mask = rolling_vol <= vol_low
    medium_vol_mask = (rolling_vol > vol_low) & (rolling_vol <= vol_high)
    high_vol_mask = rolling_vol > vol_high

    ax.fill_between(normalized_price.index, 0, normalized_price.max() * 1.1,
                     where=low_vol_mask, alpha=0.2, color='green', label='Low Volatility Period')
    ax.fill_between(normalized_price.index, 0, normalized_price.max() * 1.1,
                     where=medium_vol_mask, alpha=0.2, color='yellow', label='Medium Volatility Period')
    ax.fill_between(normalized_price.index, 0, normalized_price.max() * 1.1,
                     where=high_vol_mask, alpha=0.2, color='red', label='High Volatility Period')

    ax.plot(normalized_price.index, normalized_price, color='black', linewidth=2, label='BTC Price (Base 100)')

    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Price (Base 100)', fontsize=11, fontweight='bold')
    ax.set_title('Bitcoin Price Evolution with Volatility Risk Zones', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_volatility_risk_contribution(valid_data: pd.DataFrame, output_path: str):
    """
    Graph 12: Bitcoin Volatility Risk Contribution vs Allocation
    Shows how much of the portfolio's total volatility is attributable to Bitcoin
    for different allocation levels (inspired by "sizing bitcoin in portfolios")

    WARNING: Note: This chart shows VOLATILITY-based risk contribution.
    For CVaR-based contribution (consistent with Mean-CVaR optimization),
    see Graph 13: plot_cvar_risk_contribution.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    if 'RC_vol_BTC_ETF' not in valid_data.columns:
        print("  ⚠ RC_vol_BTC_ETF column not found, skipping volatility risk contribution chart")
        return

    btc_allocations = valid_data.index * 100  # Convert to %
    risk_contributions = valid_data['RC_vol_BTC_ETF']  # Already in %

    mask = valid_data.index > 0
    btc_allocations = btc_allocations[mask]
    risk_contributions = risk_contributions[mask]

    bars = ax.bar(btc_allocations, risk_contributions, width=0.8,
                  align='edge',  # Bars start at their X position
                  color='#FF4713', alpha=0.9, edgecolor='black', linewidth=1.5)

    bar_width = 0.8
    for i, (alloc, risk) in enumerate(zip(btc_allocations, risk_contributions)):
        ax.text(alloc + bar_width/2, risk + 0.5, f'{risk:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Allocation to Bitcoin (%)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Share of Portfolio Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_title('Bitcoin Volatility Risk Contribution vs Allocation\n(Volatility-based, not CVaR-based)',
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(0.5, 4.0)  # Frame from just before 1% to just after 3%
    ax.set_ylim(0, risk_contributions.max() * 1.10)

    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    if risk_contributions.max() > 50:
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=2, alpha=0.5,
                   label='Total Portfolio Risk (100%)')
        ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_cvar_risk_contribution(valid_data: pd.DataFrame, output_path: str):
    """
    Graph 13: Bitcoin CVaR Risk Contribution vs Allocation
    Shows how much of the portfolio's total CVaR is attributable to Bitcoin
    for different allocation levels using Euler decomposition (Acerbi-Tasche method)

    [PASS] This chart is CONSISTENT with Mean-CVaR optimization (unlike volatility-based contribution)
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    if 'RC_cvar_BTC_ETF' not in valid_data.columns:
        print("  ⚠ RC_cvar_BTC_ETF column not found, skipping CVaR risk contribution chart")
        return

    btc_allocations = valid_data.index * 100  # Convert to %
    risk_contributions = valid_data['RC_cvar_BTC_ETF']  # Already in %

    mask = valid_data.index > 0
    btc_allocations = btc_allocations[mask]
    risk_contributions = risk_contributions[mask]

    bars = ax.bar(btc_allocations, risk_contributions, width=0.8,
                  align='edge',  # Bars start at their X position
                  color='#FF6B35', alpha=0.9, edgecolor='black', linewidth=1.5)

    bar_width = 0.8
    for i, (alloc, risk) in enumerate(zip(btc_allocations, risk_contributions)):
        ax.text(alloc + bar_width/2, risk + 0.5, f'{risk:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Allocation to Bitcoin (%)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Share of Portfolio CVaR (%)', fontsize=12, fontweight='bold')
    ax.set_title('Bitcoin CVaR Risk Contribution vs Allocation\n(Euler decomposition - Consistent with Mean-CVaR optimization)',
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(0.5, 4.0)  # Frame from just before 1% to just after 3%
    ax.set_ylim(0, risk_contributions.max() * 1.10)

    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    if risk_contributions.max() > 50:
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=2, alpha=0.5,
                   label='Total Portfolio CVaR (100%)')
        ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_cumulative_performance(df_results: pd.DataFrame, prices: pd.DataFrame, output_path: str):
    """
    Graph 14: Cumulative Performance Over Time (2020-2025)
    Shows how 100€ invested would have evolved for each BTC allocation
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    valid_data = df_results[df_results['status'] == 'optimal']
    target_allocations = [0.0, 0.01, 0.02, 0.03]

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # bleu, vert, orange, rouge

    daily_returns = prices.pct_change().dropna()

    for i, wbtc in enumerate(target_allocations):
        if wbtc not in valid_data.index:
            print(f"  ⚠ Allocation {wbtc:.0%} not found in results, skipping")
            continue

        row = valid_data.loc[wbtc]
        weights = {}
        for col in prices.columns:
            weight_col = f"w_{col}"
            if weight_col in row.index:
                weights[col] = row[weight_col]
            else:
                weights[col] = 0.0

        portfolio_returns = pd.Series(0.0, index=daily_returns.index)
        for asset in prices.columns:
            portfolio_returns += daily_returns[asset] * weights[asset]

        cumulative_performance = (1 + portfolio_returns).cumprod() * 100

        label = f"{wbtc:.0%} BTC"
        ax.plot(cumulative_performance.index, cumulative_performance.values,
                linewidth=2.5, label=label, color=colors[i], alpha=0.9)

        final_value = cumulative_performance.iloc[-1]
        ax.annotate(f'{final_value:.0f}',
                   xy=(cumulative_performance.index[-1], final_value),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=colors[i],
                   ha='left', va='center')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value (Base 100 = 100€ in 2020)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Performance by Bitcoin Allocation (2020-2025)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial Investment')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_optimization_results(df: pd.DataFrame, prices: pd.DataFrame, cfg: Config,
                             output_dir: str = "btc_reserve_plots"):
    """
    Generates 14 visualization charts for optimization results.
    Each chart is saved in a separate file.

    Charts 1-11: Standard optimization and risk analysis
    Chart 12: Bitcoin Volatility Risk Contribution (volatility-based)
    Chart 13: Bitcoin CVaR Risk Contribution (CVaR-based, consistent with optimization)
    Chart 14: Cumulative Performance Over Time (2020-2025)

    Args:
        df: DataFrame with optimization results
        prices: DataFrame with historical price data
        cfg: Configuration used
        output_dir: Directory to save the charts (default: "btc_reserve_plots")
    """
    valid_data = df[df['status'] == 'optimal'].copy()
    feasible_data = valid_data[valid_data['NTL_ok'] == True].copy()

    if valid_data.empty:
        print("WARNING: No valid data to generate charts")
        return

    if not feasible_data.empty:
        idx_optimal = feasible_data['Sharpe_CVaR'].idxmax()
    else:
        idx_optimal = valid_data['cover_months'].idxmax()

    optimal_row = valid_data.loc[idx_optimal]

    output_path = ensure_output_directory(output_dir)

    print("\nGenerating optimization charts (1-6)...")

    plot_efficient_frontier(
        valid_data, feasible_data, idx_optimal, cfg,
        os.path.join(output_path, "btc_01_efficient_frontier.png")
    )

    plot_portfolio_composition(
        valid_data,
        os.path.join(output_path, "btc_02_portfolio_composition.png")
    )

    plot_sharpe_cvar(
        valid_data, feasible_data, idx_optimal,
        os.path.join(output_path, "btc_03_sharpe_cvar.png")
    )

    plot_return_vs_btc(
        valid_data, idx_optimal, optimal_row,
        os.path.join(output_path, "btc_04_return_vs_btc.png")
    )

    plot_cvar_vs_btc(
        valid_data, idx_optimal, optimal_row,
        os.path.join(output_path, "btc_05_cvar_vs_btc.png")
    )

    plot_combined_metrics(
        valid_data, idx_optimal,
        os.path.join(output_path, "btc_06_combined_metrics.png")
    )

    print("\nGenerating Bitcoin risk analysis charts (7-11)...")

    plot_btc_drawdowns(
        prices,
        os.path.join(output_path, "btc_07_drawdowns.png")
    )

    plot_return_distributions(
        prices,
        os.path.join(output_path, "btc_08_monthly_returns_dist.png")
    )

    plot_volatility_comparison(
        prices,
        os.path.join(output_path, "btc_09_volatility_comparison.png")
    )

    plot_worst_months(
        prices,
        os.path.join(output_path, "btc_10_worst_month.png")
    )

    plot_btc_price_with_risk_zones(
        prices,
        os.path.join(output_path, "btc_11_btc_price_risk_zones.png")
    )

    print("\nGenerating risk contribution analysis charts (12-13)...")

    plot_volatility_risk_contribution(
        valid_data,
        os.path.join(output_path, "btc_12_volatility_risk_contribution.png")
    )

    plot_cvar_risk_contribution(
        valid_data,
        os.path.join(output_path, "btc_13_cvar_risk_contribution.png")
    )

    print("\nGenerating cumulative performance chart (14)...")

    plot_cumulative_performance(
        df,
        prices,
        os.path.join(output_path, "btc_14_cumulative_performance.png")
    )

    print(f"\n[PASS] All 14 charts saved successfully in: {output_path}/")



def test_gaussian_consistency():
    """
    Gaussian consistency test: verifies that on an IID Gaussian series,
    the 12m CVaR by additive bootstrap (blocks=1) corresponds to the theoretical formula
    CVaR_12m = -12μ + κσ√12 with κ = φ(z_0.99)/(1-0.99).
    Tolerance: relative error ≤ 5%
    """
    print("\nTest 1/3: Gaussian consistency")

    np.random.seed(42)
    n_months = 60
    monthly_returns = pd.Series(np.random.normal(0.005, 0.02, n_months))

    risk_bootstrap = compute_conditional_value_at_risk_12m_block_bootstrap(
        monthly_returns,
        confidence_level=0.99,
        number_of_paths=50000,  # Increased to reduce Monte Carlo error
        block_length_min=1,
        block_length_max=1,
        random_seed=42,
        aggregation="additive"  # Additive mode for Gaussian validation
    )
    cvar_bootstrap = risk_bootstrap["conditional_value_at_risk_12m"]

    from scipy.stats import norm
    z = norm.ppf(0.99)
    kappa = norm.pdf(z) / (1 - 0.99)  # ≈ 2.665
    mu, sigma = monthly_returns.mean(), monthly_returns.std(ddof=1)
    cvar_12m_theory = -12*mu + kappa * sigma * np.sqrt(12)

    relative_error = abs(cvar_bootstrap - cvar_12m_theory) / max(1e-12, cvar_12m_theory)

    passed = relative_error <= 0.05
    status = "PASS" if passed else "FAIL"

    print(f"   12m CVaR (additive bootstrap):  {cvar_bootstrap:.4f}")
    print(f"   12m CVaR (Gaussian formula): {cvar_12m_theory:.4f}")
    print(f"   Relative error:                 {relative_error:.2%}")
    print(f"   Result: {status} (tolerance: 5%)")

    return passed


def test_reproducibility():
    """
    Reproducibility test: same input + same random_seed → same output
    """
    print("\nTest 2/3: Reproducibility")

    np.random.seed(123)
    monthly_returns = pd.Series(np.random.normal(0.005, 0.02, 50))

    risk_1 = compute_conditional_value_at_risk_12m_block_bootstrap(
        monthly_returns,
        confidence_level=0.99,
        number_of_paths=5000,
        random_seed=99
    )

    risk_2 = compute_conditional_value_at_risk_12m_block_bootstrap(
        monthly_returns,
        confidence_level=0.99,
        number_of_paths=5000,
        random_seed=99
    )

    cvar_identical = risk_1["conditional_value_at_risk_12m"] == risk_2["conditional_value_at_risk_12m"]
    var_identical = risk_1["value_at_risk_12m"] == risk_2["value_at_risk_12m"]

    passed = cvar_identical and var_identical
    status = "PASS" if passed else "FAIL"

    print(f"   CVaR run 1: {risk_1['conditional_value_at_risk_12m']:.6f}")
    print(f"   CVaR run 2: {risk_2['conditional_value_at_risk_12m']:.6f}")
    print(f"   Identical: {cvar_identical and var_identical}")
    print(f"   Result: {status}")

    return passed


def test_performance():
    """
    Performance test: 50K paths must execute in < 30 seconds
    """
    print("\nTest 3/3: Performance")

    np.random.seed(456)
    monthly_returns = pd.Series(np.random.normal(0.005, 0.02, 50))

    start_time = time.time()

    risk = compute_conditional_value_at_risk_12m_block_bootstrap(
        monthly_returns,
        confidence_level=0.99,
        number_of_paths=50000,
        block_length_min=3,
        block_length_max=6,
        random_seed=42
    )

    elapsed_time = time.time() - start_time

    passed = elapsed_time < 30.0
    status = "PASS" if passed else "FAIL"

    print(f"   Simulated paths: 50,000")
    print(f"   Execution time: {elapsed_time:.2f}s")
    print(f"   Limit: 30.0s")
    print(f"   Result: {status}")

    return passed


def run_validation_tests():
    """
    Executes all validation tests
    """
    print("\n" + "="*80)
    print(" "*25 + "VALIDATION TESTS")
    print("="*80)

    results = {
        "gaussian_consistency": test_gaussian_consistency(),
        "reproducibility": test_reproducibility(),
        "performance": test_performance()
    }

    total = len(results)
    passed = sum(results.values())

    print("\n" + "="*80)
    print(" "*30 + "TEST SUMMARY")
    print("="*80)
    print(f"   Tests passed: {passed}/{total}")

    if passed == total:
        print("   [PASS] All tests passed!")
    else:
        print(f"   WARNING: {total - passed} test(s) failed")
        failed_tests = [name for name, result in results.items() if not result]
        print(f"   Failed tests: {', '.join(failed_tests)}")

    print("="*80)

    return passed == total



if __name__ == "__main__":
    RUN_TESTS = True  # Set to False to skip tests

    if RUN_TESTS:
        tests_passed = run_validation_tests()
        if not tests_passed:
            print("\nWARNING: Some tests failed. Continue with analysis? (y/n)")
            print("   Automatic mode: continuing...")

    cfg = Config(
        beta=0.99,                    # CVaR at 99%
        risk_priority=0.75,           # Risk priority
        btc_tiers=tuple(np.arange(0.0, 0.0301, 0.001)),  # 101 simulations (0-3%, step 0.1%)
        weight_bounds=(0, 1),         # Long-only
        verbose=False
    )

    try:
        df_results, prices = run_pypfopt_pipeline(start="2020-01-01", end="2025-10-08", cfg=cfg)

        display_results_pypfopt(df_results, cfg)

        print("\n" + "="*80)
        print("Generating visualization charts...")
        print("="*80)
        plot_optimization_results(df_results, prices, cfg, output_dir="btc_reserve_plots")

        print("\nPyPortfolioOpt analysis completed successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()



