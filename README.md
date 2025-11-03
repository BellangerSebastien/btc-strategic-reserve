# BTC Strategic Reserve - Portfolio Optimization Framework

Mean-CVaR portfolio optimization framework for institutional Bitcoin treasury allocation using PyPortfolioOpt.

---

## Overview

This framework implements a quantitative approach to Bitcoin treasury management, optimizing portfolio allocation across traditional fixed-income instruments (Money Market Funds, Ultra-Short Bonds, Investment-Grade Short Bonds) and Bitcoin exposure via ETF (iShares Bitcoin Trust - IBIT).

The tool employs **Mean-CVaR optimization** with advanced risk estimation techniques to determine optimal Bitcoin allocation levels while maintaining strict liquidity constraints.

---

## Features

### Core Capabilities
- **Mean-CVaR Portfolio Optimization** using PyPortfolioOpt's `EfficientCVaR`
- **Multi-tier BTC allocation analysis** (0% to 3% in 0.1% increments)
- **12-month risk estimation** via block bootstrap methodology
- **NTL-at-Risk liquidity analysis** ensuring minimum coverage requirements
- **Comprehensive visualization suite** (14 professional charts)

### Risk Estimation Methods
1. **Block Bootstrap** (default): Preserves autocorrelation and regime dependencies
2. **Extreme Value Theory (EVT)**: Peaks Over Threshold approach for tail risk

### Asset Universe
- **MMF**: Money Market Fund (40-55% bounds)
- **ULTRA_SHORT**: Ultra-Short Duration Bonds (20-30% bounds)
- **IG_SHORT**: Investment-Grade Short-Term Bonds (20-30% bounds)
- **BTC_ETF**: iShares Bitcoin Trust - IBIT (0-3% analyzed)

---

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
yfinance>=0.2.0
pypfopt>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/[your-username]/btc-strategic-reserve.git
cd btc-strategic-reserve
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas yfinance pypfopt matplotlib seaborn scipy
```

---

## Usage

### Basic Execution
```bash
python btc_reserve_pypfopt.py
```

### Configuration Parameters

Modify the `Config` class to adjust optimization parameters:

```python
cfg = Config(
    beta=0.99,                    # CVaR confidence level (99%)
    risk_priority=0.75,           # Risk vs return priority
    btc_tiers=(0.0, 0.01, 0.02, 0.03),  # BTC allocation levels to test
    weight_bounds=(0, 1),         # Long-only constraint
    verbose=False                 # Detailed optimizer output
)
```

### Risk Estimation Configuration

```python
risk_config = RiskEstimationConfig(
    confidence_level=0.99,              # 99% confidence
    method="block_bootstrap",           # or "extreme_value_theory"
    number_of_paths=50000,              # Monte Carlo paths
    block_length_min=3,                 # Min block size
    block_length_max=6,                 # Max block size
    random_seed=42                      # Reproducibility
)
```

---

## Methodology

### Mean-CVaR Optimization

The framework uses **Conditional Value at Risk (CVaR)** as the risk measure, optimizing for:
```
maximize: Expected Return - λ × CVaR_β
```

Where:
- **CVaR_β**: Expected loss in the worst β% of scenarios (β = 99%)
- **λ**: Risk aversion parameter (derived from `risk_priority`)

### Block Bootstrap Risk Estimation

12-month risk metrics are estimated using block bootstrap:

1. Convert monthly returns to log-returns
2. Sample consecutive blocks of random length [3-6 months]
3. Concatenate blocks to create 12-month paths
4. Compound returns: `r_12m = exp(Σ log_returns) - 1`
5. Calculate VaR and CVaR from simulated loss distribution

This approach preserves:
- **Autocorrelation** in returns
- **Regime persistence** (bull/bear market clustering)
- **Non-normality** of return distributions

### NTL-at-Risk Analysis

Liquidity constraints ensure portfolio resilience:

```
NTL_post_stress = NTL_current - (CVaR_monthly × Portfolio_Size)
Coverage_Months = NTL_post_stress / Monthly_Outflows
```

Requirement: `Coverage_Months ≥ 3`

---

## Output

### Console Results

1. **Validation Tests**
   - Gaussian consistency test
   - Reproducibility verification
   - Performance benchmarking

2. **Optimization Table**
   - Expected annualized returns
   - Monthly and 12-month CVaR/VaR
   - Sharpe-CVaR ratios
   - NTL impact and coverage
   - Optimal portfolio weights

3. **Liquidity Analysis**
   - Feasibility check
   - Optimal allocation recommendation
   - Risk-return tradeoffs

### Visualizations (14 Charts)

Saved in `btc_reserve_plots/`:

1. **btc_01_efficient_frontier.png** - Mean-CVaR efficient frontier
2. **btc_02_portfolio_composition.png** - Asset allocation stacked area
3. **btc_03_sharpe_cvar.png** - Risk-adjusted performance bars
4. **btc_04_return_vs_btc.png** - Expected return vs BTC allocation
5. **btc_05_cvar_vs_btc.png** - CVaR progression
6. **btc_06_combined_metrics.png** - Multi-axis performance view
7. **btc_07_drawdowns.png** - Bitcoin maximum drawdown timeline
8. **btc_08_monthly_returns_dist.png** - Return distribution comparison
9. **btc_09_volatility_comparison.png** - Annualized volatility bars
10. **btc_10_worst_month.png** - Worst historical monthly returns
11. **btc_11_btc_price_risk_zones.png** - BTC price with volatility zones
12. **btc_12_volatility_risk_contribution.png** - Volatility-based risk attribution
13. **btc_13_cvar_risk_contribution.png** - CVaR-based risk attribution (Euler decomposition)
14. **btc_14_cumulative_performance.png** - Historical performance backtest

---

## Technical Details

### Data Source
- **Provider**: Yahoo Finance (via `yfinance`)
- **Period**: 2020-01-01 to present
- **Frequency**: Daily prices, resampled to monthly
- **Tickers**:
  - CSH.PA (MMF)
  - IS3M.DE (ULTRA_SHORT)
  - SE15.L (IG_SHORT)
  - BTCE.DE (BTC_ETF)

### Risk Metrics

**Monthly Risk**:
- CVaR_monthly: Portfolio CVaR over 1 month
- VaR_monthly: Portfolio VaR over 1 month

**12-Month Compounded Risk**:
- CVaR_12m: Bootstrapped 12-month CVaR (NO √12 scaling)
- VaR_12m: Bootstrapped 12-month VaR
- Sharpe_CVaR_12m: Return / CVaR_12m ratio

### Validation Tests

1. **Gaussian Consistency**: Verifies that block bootstrap matches theoretical CVaR under IID Gaussian assumptions (tolerance: 5%)
2. **Reproducibility**: Ensures identical results with same random seed
3. **Performance**: Confirms 50,000 paths execute in < 30 seconds

---

## Example Results

```
OPTIMAL ALLOCATION [PASS] WITH NTL CONSTRAINT: 2.10% BTC
   • Expected annualized return: 2.47%
   • CVaR (99%) monthly: 1.12%
   • CVaR (99%) 12-month compounded: 3.45%
   • VaR (99%) 12-month compounded: 2.81%
   • Sharpe-CVaR ratio (12m): 0.717

NTL IMPACT
   • Monthly stress loss: 11.2M€
   • NTL post-stress: 1989M€
   • Coverage post-stress: 4.0 months
   • Constraint satisfied: True
```

---

## Configuration Constants

```python
POOL_EUR = 1_000_000_000           # Portfolio size (1B EUR)
NTL_EUR = 2_000_000_000            # Net Treasury Liquidity (2B EUR)
MONTHLY_OUTFLOWS_EUR = 500_000_000 # Monthly cash needs (500M EUR)
MIN_COVER_MONTHS = 3               # Minimum coverage requirement
```

---

## License

This project is provided for educational and research purposes.

---

## Author

Sébastien Bellanger

---

## References

- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21-42.
- Acerbi, C., & Tasche, D. (2002). Expected shortfall: A natural coherent alternative to value at risk. *Economic Notes*, 31(2), 379-388.
- PyPortfolioOpt: [https://github.com/robertmartin8/PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)

---

## Notes

- **CVaR** is also known as **Expected Shortfall (ES)** or **Tail Conditional Expectation (TCE)**
- The 12-month risk metrics use **compounded returns**, not √12 scaling
- Block bootstrap preserves temporal dependencies unlike standard bootstrap
- NTL-at-Risk uses **monthly CVaR** to assess liquidity stress scenarios
