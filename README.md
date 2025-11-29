# Regime-Switching XGBoost Stock Selection Model

A quantitative equity strategy that uses economic regime classification combined with machine learning to identify stocks likely to outperform over the next 12 months. The model trains separate XGBoost classifiers for different macroeconomic environments and applies canonical factor investing principles (value, quality, momentum predictors) validated by academic research.

**Key Features:**
- BigQuery data warehouse with 640K+ company-quarter observations (2000-2025)
- Automated weekly data pipelines via GitHub Actions
- PCA-driven economic regime classification with K-means clustering (3 regimes: Growth, Slow Growth, Recession)
- Factor-based feature engineering aligned with academic research (value, quality, momentum)
- Regime-specific XGBoost ensemble classifiers
- Forward-step k-fold cross-validation with 1-year embargo periods


## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Data Requirements](#data-requirements)
- [Model Methodology](#model-methodology)
- [Feature Engineering](#feature-engineering)
- [Future Enhancements](#future-enhancements)
- [Disclaimer](#disclaimer)
- [License](#license)

## Overview

This project implements a regime-switching quantitative stock selection framework that adapts its predictive models to changing macroeconomic conditions. Rather than applying a single model to all market environments, the system identifies the current economic regime and applies the appropriate regime-trained classifier.

**Core Innovation:** Two-stage adaptive modeling:
1. **Regime-Level**: Classify macroeconomic environment using unsupervised learning (PCA + K-means)
2. **Stock-Level**: Apply regime-specific XGBoost models trained on historical factor performance within that regime

**Business Objective:** Generate consistent alpha by adapting stock selection criteria to prevailing economic conditions, using factors (value, quality, momentum) proven effective in academic research.

## Model Architecture
```
Economic Data → PCA Reduction → K-Means → Feature Engineering → Regime-Specific → Binary
     ↓              ↓              ↓            ↓                  XGBoost          Classification
Macro Variables  2 Principal   3 Economic   Value/Quality/      Classifiers      Investment Grade
(GDP MA,         Components    Regimes      Momentum Factors    (separate per    vs Non-Grade
Fed Funds MA,                 (Growth,      + Industry          regime)
Yield Curve)                   Slow Growth, Rankings
                              Recession)
```

### Model Components

1. **Economic Regime Classifier**: 
   - **Unsupervised learning approach:** PCA dimensionality reduction + K-means clustering
   - **Input:** 16 macroeconomic variables from FRED (GDP, Fed Funds Rate, Yield Curve, Unemployment, Housing Starts, Industrial Production)
   - **PCA:** Reduces to 2 principal components, trained on 1985-2005 period
   - **K-means:** 3 clusters representing Growth, Slow Growth, and Recession regimes
   - **Application:** Quarterly regime classification from 2005-present with 1-quarter lag
   
2. **Factor-Based Feature Engineering**: 
   - **Value factors:** EV/EBIT, EV/EBITDA, P/E, P/FCF, P/B, P/TB percentiles
   - **Quality factors:** ROE, Piotroski F-Score, cash flow persistence, industry rankings
   - **Momentum factors:** 3-month and 12-month price momentum
   - **Additional signals:** Superinvestor holdings, insider trading activity
   - All features calculated as industry-relative percentiles and company-specific historical percentiles

3. **XGBoost Ensemble Models**: 
   - **Separate models per regime:** 3 regimes × multiple validation folds = 12+ models
   - **Model type:** Boosted Tree Classifier (binary classification)
   - **Target variable:** Percentile ranking for 12 month forward returns bucketed into 5 distinct buckets (top 20th percentile, next 20th, etc...)
   - **Validation:** Forward-step k-fold with expanding training windows, 1 year embargo periods between training and validation

## Data Requirements

**Primary Data Sources:**
- **SEC EDGAR filings:** 10-K and 10-Q fundamental data (2000-2025)
- **FRED Economic Data:** Macroeconomic indicators for regime classification
- **Yahoo Finance:** Daily stock prices for momentum calculations (In progress, currently using quarterly prices)
- **Dataroma:** Superinvestor holdings data (13F filings aggregated)
- **SEC Form 4:** Insider trading data (In progress)

**Data Infrastructure:**
- **Storage:** Google BigQuery data warehouse
- **Volume:** 640K+ company-quarter observations from NASDAQ and NYSE listed companies (>$300M market cap)
- **Update frequency:** Daily automated pipelines via GitHub Actions
- **Cost:** ~$1/month (BigQuery storage + compute)

**Data Quality Requirements:**
- Minimum 5-year history for percentile calculations
- Price > 0, next_year_price > 0, market_cap > 0
- Winsorization at 2nd/98th percentiles for all relevant predictors to limit outlier impact

## Model Methodology

### Economic Regime Classification

**Approach:** Unsupervised learning (PCA + K-means)

**Step 1 - Dimensionality Reduction (PCA):**
- **Input variables:** 16 macroeconomic indicators, winsorized at 5th-95th percentiles:
  - GDP 2-quarter moving average
  - Federal Funds Rate year-over-year change (2Q MA)
  - Yield Curve Spread (2Q MA)
  - Unemployment rate, housing starts, industrial production trends and volatilities

- **PCA configuration:**
  - Standard scaler normalization
  - Reduction to 2 principal components
  - Training period: 1985-2005 (pre-financial crisis era)
  - PC scores additionally winsorized at ±1.96 to prevent outlier influence

**Step 2 - Regime Clustering (K-means):**
- **Algorithm:** K-means++ initialization
- **Number of clusters:** 3 economic regimes
  - **Regime 1 (Slow Growth):** Low growth, moderate volatility
  - **Regime 2 (Recession):** Negative growth, high volatility
  - **Regime 3 (Growth):** Positive growth, stable conditions
- **Parameters:**
  - max_iterations = 50
  - distance_type = EUCLIDEAN
  - standardize_features = TRUE

**Regime Application:**
- Applied quarterly from 2000-present
- **1-quarter lag** to avoid look-ahead bias (use Q1 regime to predict Q2 stocks)
- Regime labels stored in BigQuery table for model routing

### Feature Engineering

**Top Features by Importance (Validated Against Academic Research):**

The model's feature importance rankings align remarkably well with established factor investing literature:

1. **ev_ebit_winsorized** - Enterprise Value to EBIT (primary value factor)
2. **price_momentum_12mo_winsorized** - 12-month price momentum (canonical momentum factor)
3. **pe_winsorized** - Price to Earnings ratio (traditional value)
4. **ev_ebitda_winsorized** - Enterprise Value to EBITDA (value factor)
5. **market_cap_buckets** - Size factor (small-cap premium)
6. **industry_id** - Sector classification
7. **revenue_cagr_10_winsorized** - 10-year revenue growth (quality/growth)
8. **momentum_12mo_industry_ranking** - Relative momentum within industry
9. **price_momentum_3mo_winsorized** - Short-term momentum
10. **ev_ebitda_over_p50_industry_winsorized** - Industry-relative valuation
11. **ptb_winsorized** - Price to Tangible Book (value factor)
12. **ev_ebit_percentile_industry** - Industry-relative EV/EBIT ranking
13. **pfcf_winsorized** - Price to Free Cash Flow (value factor)
14. **piotroski_f_score** - 9-point fundamental quality score (Piotroski 2000)
15. **return_on_equity_winsorized** - Profitability factor (quality)
16. **gross_profit_to_total_assets_industry_ranking** - Asset efficiency (quality)
17. **roic_industry_ranking** - Return on Invested Capital ranking (quality)
18. **growth_consistency_score** - Revenue/earnings stability

This feature hierarchy validates the model's foundation in established factor research rather than data mining artifacts.

**Value Factors:**
All valuation metrics calculated as both winsorized absolute values and industry-relative percentiles:
- **EV/EBIT** - Enterprise value to operating income (top importance)
- **EV/EBITDA** - Enterprise value to EBITDA
- **P/E ratio** - Price to earnings
- **P/FCF** - Price to free cash flow  
- **P/B** - Price to book value
- **P/TB** - Price to tangible book value
- **Industry-relative comparisons** - Current value vs industry P50 thresholds

**Quality Factors:**
- **Piotroski F-Score** - 9-point fundamental strength score:
  - Profitability: ROA > 0, Operating cash flow > 0, FCF > Net Income
  - Leverage/Liquidity: Decreasing leverage, improving current ratio, no dilution
  - Operating Efficiency: Improving margin, improving asset turnover, revenue growth
- **Return on Equity (ROE)** - Company-specific and industry-relative rankings
- **ROE trend** - 5-year trend in profitability
- **ROIC industry ranking** - Return on invested capital vs peers
- **Revenue/EPS CAGR (10-year)** - Long-term sustainable growth rates
- **Cash flow persistence** - Consistency of positive FCF over 5 years
- **Gross profit to assets** - Margin quality indicator
- **Growth consistency score** - Stability of revenue and earnings growth

**Momentum Factors:**
- **12-month price momentum** - Canonical momentum factor (top 2 importance)
- **3-month price momentum** - Short-term trend confirmation
- **12-month momentum industry ranking** - Relative strength within sector

**Size Factor:**
- **Market cap buckets** - Large-cap, Mid-cap, Small-cap, Micro-cap classification
- Controls for size effect documented in Fama-French research

**Additional Alpha Signals:**
- **Superinvestor holdings %** - Ownership by prominent value investors (13F filings)
- **Change in superinvestor holdings** - Recent insider buying/selling activity
- **Total distribution yield** - Dividends + buybacks % of market cap
- **Industry classification** - Sector-specific patterns

**Feature Transformations:**
- **Winsorization:** All continuous features at 2nd/98th percentiles per quarter to limit outlier impact
- **Industry-relative percentiles:** Current value vs industry distribution (rolling 5-year window)
- **Historical percentiles:** Current value vs company's own history (rolling 5-year window)
- **Standardization:** Z-score normalization for features with wide ranges
- **Missing data imputation:** Sector medians for missing fundamental values


### XGBoost Model Configuration

**Model Type:** Boosted Tree Classifier (Binary Classification)

**Target Variable:** 
- `investment_grade` (5-level classification)
- Defined by forward return thresholds and quality screening
- Separates high-potential stocks from the broader universe

**Hyperparameters** (regime-specific models):
```yaml
model_type: BOOSTED_TREE_CLASSIFIER
num_parallel_tree: 6
max_tree_depth: 5
max_iterations: 400
learn_rate: 0.03
subsample: 0.7
l1_reg: 1.0 - 2.0  # Varies by regime
l2_reg: 0.5 - 2.0  # Varies by regime
min_rel_progress: 0.0001
data_split_method: CUSTOM  # Uses temporal split
```

## Future Enhancements

**Regime Modeling:**
- Experiment with 4-5 clusters for finer economic distinctions
- Update PCA training period to include 2008 financial crisis and COVID
- Add regime transition smoothing to reduce whipsaw
- Test regime-conditional factor weighting

**Feature Engineering:**
- Expand quality metrics (asset turnover, working capital management)
- Add analyst sentiment and revision features
- Add insider trading data / feature engineering
- Ingest 10k/10q raw text for feature engineering and sentiment analyses
- Incorporate options-implied volatility as risk measure
- Test alternative momentum definitions (6-month, 9-month)

**Model Architecture:**
- Implement stacking ensemble across regime models
- Add calibration layer for probability scores
- Test regime probability weighting vs hard regime assignment
- Explore gradient boosting alternatives (LightGBM, CatBoost)

**Validation & Risk Management:**
- Implement purged k-fold cross-validation (López de Prado)
- Build transaction cost simulator with market impact models
- Refine development of position sizing logic using Kelly criterion or risk parity

## Disclaimer

**Important Risk Disclosure:** This model is for educational and research purposes only. Past performance does not guarantee future results. The model may experience significant losses, particularly during regime transitions or unprecedented market conditions not represented in historical data.

**Not Investment Advice:** This software is not investment advice or a recommendation to buy/sell securities. Users must conduct their own due diligence and consult qualified financial advisors before making investment decisions.

**Model Limitations:**
- **Backtest Overfitting:** Historical performance likely overstates future results due to data mining, look-ahead bias, and survivorship bias
- **Regime Detection Lag:** Economic regime classification operates on quarterly lag and may miss rapid market transitions
- **Transaction Costs:** Backtest results do not include trading costs, market impact, slippage, or implementation shortfall
- **Data Quality Dependency:** Model performance critically depends on accurate, timely fundamental and macroeconomic data
- **Limited Historical Coverage:** Only tested on 2000-2025 period, may not generalize to different market regimes
- **No Professional Oversight:** Model built independently without institutional risk management or compliance review

**Technical Limitations:**
- Requires substantial BigQuery compute resources for training and scoring
- 5-year minimum history requirement excludes many stocks from universe
- Binary classification approach loses granularity vs continuous probability scores
- Fixed industry taxonomy may become stale as sector definitions evolve
- No built-in position sizing, portfolio construction, or risk management logic
