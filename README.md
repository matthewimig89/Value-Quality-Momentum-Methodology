# Value-Quality Stock Selection Model

A quantitative stock selection framework that combines economic regime classification with adaptive valuation metrics to identify investment-grade stocks. The model uses PCA-based regime detection and trains separate XGBoost classifiers for different economic environments, dynamically selecting the most predictive valuation metric for each company.

**Key Features:**
- PCA-driven economic regime classification with K-means clustering (3 regimes)
- Adaptive valuation metric selection based on historical alpha correlation
- Company-specific and industry-relative percentile rankings (5-year rolling windows)
- XGBoost ensemble classifiers with regime-specific training (4 separate models per fold)
- Forward-step validation with expanding training windows and out-of-time testing
- Binary classification approach (investment-grade vs non-investment-grade stocks)

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Data Requirements](#data-requirements)
- [Installation \& Setup](#installation--setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Methodology](#model-methodology)
- [Feature Engineering](#feature-engineering)
- [Risk Management](#risk-management)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

## Overview

This project implements a multi-stage quantitative stock selection model that adapts both its valuation methodology and predictive approach to different economic environments. Rather than applying a single valuation metric to all companies, the model identifies each company's most predictive valuation metric based on historical alpha correlation, then applies regime-specific models to classify stocks.

**Core Innovation:** The model combines two levels of adaptation:
1. **Company-Level**: Selects the valuation metric with strongest historical correlation to forward returns for each company
2. **Regime-Level**: Trains separate classification models for each economic environment

**Business Objective:** Generate consistent alpha by dynamically adjusting both the valuation methodology and the classification approach based on company-specific historical patterns and prevailing economic conditions.

## Model Architecture

```
Economic Data → PCA Reduction → K-Means → Feature Engineering → Regime-Specific → Binary
     ↓              ↓              ↓            ↓                  XGBoost          Classification
Macro Variables  2 Principal   3 Economic   Adaptive Valuation   Classifiers      Investment Grade
(GDP MA,         Components    Regimes      Metric Selection     (4 per fold)     vs Non-Grade
Fed Funds MA,                              (P/E, P/B, P/FCF,
Yield Curve)                               EV/EBIT, EV/EBITDA)
```

### Model Components

1. **Economic Regime Classifier**: 
   - PCA dimensionality reduction (16 macro variables → 2 principal components)
   - K-means clustering on principal components (k=3 regimes)
   - Training period: 1985-2005, applied 2000-present
   
2. **Adaptive Valuation Framework**:
   - Evaluates 6 valuation metrics per company: P/E, P/B, P/TB, P/FCF, EV/EBIT, EV/EBITDA
   - Calculates alpha correlation for each metric using historical 20th-30th percentile thresholds
   - Selects "best valuation measure" with strongest predictive power
   
3. **Feature Engineering Pipeline**: 
   - Company-specific percentile rankings (5-year rolling window)
   - Industry-relative valuation metrics (vs P25, P50, P75 quantiles)
   - Winsorization at 5th/95th percentiles
   - Alpha correlation metrics for dynamic valuation selection

4. **Ensemble Classification Models**: 
   - Binary target: Investment-grade stocks (based on forward return thresholds)
   - 4 regime-specific XGBoost classifier models per validation fold
   - Separate models for classification and regression variants
   
5. **Validation Framework**: 
   - Forward-step validation with expanding training windows
   - Custom data splits using k-fold temporal partitions
   - Out-of-time validation on future periods

## Data Requirements

### Primary Data Sources

**Market Data:**
- Daily equity prices and volumes (comprehensive US equity universe)
- Market capitalization data (minimum $2B threshold recommended)
- Industry classifications (custom 25+ industry taxonomy)
- Quarterly rebalancing dates

**Fundamental Data:**
- Quarterly financial statements (Balance Sheet, Income Statement, Cash Flow)
- At least 5 years of historical data required for percentile calculations
- Valuation metrics: P/E, P/B, P/TB, P/FCF, EV/EBIT, EV/EBITDA
- Quality metrics: ROE, profitability indicators, growth rates

**Macroeconomic Data (FRED sources):**
- **Primary Variables (for PCA)**:
  - GDP growth (2-quarter moving average)
  - Federal Funds Rate year-over-year change (2-quarter moving average)
  - Yield Curve Spread (10Y-2Y, 2-quarter moving average)

- **Additional Variables (for feature engineering)**:
  - Unemployment rate and volatility measures
  - Housing starts trend indicators
  - Industrial production metrics
  - Inflation data
  
### Storage Requirements

- **BigQuery Dataset Size**: ~500GB for 10-year history with comprehensive feature set
- **Processing Memory**: 16GB RAM minimum for model training queries
- **Query Cost Estimate**: $50-100/month for regular backtesting and retraining

## Installation & Setup

### Prerequisites

```bash
# Python environment (for data ingestion and orchestration)
pip install google-cloud-bigquery
pip install pandas numpy

# BigQuery authentication
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### BigQuery Setup

1. **Enable BigQuery API** in Google Cloud Console

2. **Create dataset** for model artifacts:
```sql
CREATE SCHEMA IF NOT EXISTS `your-project.value_quality_model`;
CREATE SCHEMA IF NOT EXISTS `your-project.macroeconomic_data`;
```

3. **Set up service account** with BigQuery Data Editor and BigQuery ML User permissions

4. **Configure billing** with sufficient quota for ML operations (BQML models can be resource-intensive)

### Data Pipeline Setup

```sql
-- Load your fundamental and market data into base tables
-- Create economic indicators table
CREATE TABLE `your-project.macroeconomic_data.macro_indicators_value_quality_model_quarterly` AS
SELECT * FROM your_macro_data_source;

-- Verify data structure matches expected schema
```

## Usage

### Quick Start

The model consists of three main execution stages that must be run sequentially:

### Running Full Pipeline

**1. Build Economic Features & Train Regime Classifier**
```bash
# First, build the economic features from raw macro data
bq query --use_legacy_sql=false < build-economic-features.sql

# Then train the PCA and K-means regime classifier
bq query --use_legacy_sql=false < train-economic-regime-classifier.sql
```

**2. Feature Engineering & Model Training**
```bash
# Build comprehensive stock features with adaptive valuation metrics
bq query --use_legacy_sql=false < value-quality-stock-return-feature-extraction-model-build.sql
```

**3. Cross-Validation Training**
```bash
# Train all regime-specific models with forward-step validation
# This creates 4 regime models × multiple folds × classification & regression variants
bq query --use_legacy_sql=false < XGBoost-5Fold-Forward-Step-Validation-Model-Build.sql
```

**4. Generate Predictions**
```bash
# Predict current economic regime
bq query --use_legacy_sql=false < predict-economic-regime.sql

# Generate stock classifications using appropriate regime model
# (Query the relevant model based on predicted regime)
```

### Python Integration Example

```python
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

# Get current economic regime
regime_query = """
SELECT DISTINCT 
  quarter_end,
  CENTROID_ID as regime,
  principal_component_1,
  principal_component_2
FROM ML.PREDICT(
  MODEL `your-project.macroeconomic_data.macro_regime_kmeans`,
  (SELECT * FROM `your-project.macroeconomic_data.principal_components_current`)
)
ORDER BY quarter_end DESC
LIMIT 1
"""
current_regime = client.query(regime_query).to_dataframe()['regime'].iloc[0]

# Get stock predictions for current regime
predictions_query = f"""
SELECT 
  company_id,
  symbol,
  predicted_investment_grade,
  predicted_investment_grade_probs[OFFSET(1)].prob as investment_grade_probability
FROM ML.PREDICT(
  MODEL `your-project.value_quality_model.value_quality_model_k{current_regime}_1`,
  (SELECT * EXCEPT(period_end_date) 
   FROM `your-project.value_quality_model.current_stock_features`)
)
WHERE predicted_investment_grade = TRUE
ORDER BY investment_grade_probability DESC
LIMIT 50
"""

top_stocks = client.query(predictions_query).to_dataframe()
print(f"Current Regime: {current_regime}")
print(f"\nTop 10 Investment-Grade Stock Candidates:\n{top_stocks.head(10)}")
```

## File Structure

```
├── README.md
├── train-economic-regime-classifier.sql    # PCA + K-means regime classification
├── predict-economic-regime.sql             # Apply trained regime classifier
├── build-economic-features.sql             # Prepare macro features
├── value-quality-stock-return-feature-extraction-model-build.sql  # Core feature engineering
└── XGBoost-5Fold-Forward-Step-Validation-Model-Build.sql         # Train all models
```

## Model Methodology

### Economic Regime Classification

**Approach**: PCA + K-means unsupervised learning

**Step 1 - Dimensionality Reduction**:
- **Input Variables** (16 macroeconomic indicators, winsorized):
  - GDP 2-quarter moving average (winsorized 5th-95th percentile)
  - Federal Funds Rate YoY change 2Q MA (winsorized)
  - Yield Curve Spread 2Q MA (winsorized)
  - Plus additional volatility and trend measures

- **PCA Configuration**:
  - Standard scaler normalization
  - Reduction to 2 principal components
  - Training period: 1985-2005
  - Additional winsorization on PC scores at ±1.96 (prevent outlier influence)

**Step 2 - Regime Clustering**:
- **K-means Algorithm**:
  - num_clusters = 3 (not 4 as in traditional models)
  - kmeans_init_method = 'KMEANS_PLUS_PLUS'
  - max_iterations = 50
  - early_stop = TRUE
  - distance_type = 'EUCLIDEAN'

**Regime Application**:
- Applied to quarterly data from 2000-present
- Lagged 1 quarter to avoid look-ahead bias

### Adaptive Valuation Framework

**Core Concept**: Different valuation metrics predict returns better for different companies. Rather than applying P/E uniformly, the model identifies which metric historically worked best for each company.

**Valuation Metrics Evaluated**:
1. Price-to-Earnings (P/E)
2. Price-to-Book (P/B)
3. Price-to-Tangible Book (P/TB)
4. Price-to-Free Cash Flow (P/FCF)
5. Enterprise Value / EBIT
6. Enterprise Value / EBITDA

**Alpha Correlation Methodology**:
- Calculate 20th and 30th percentile thresholds for each metric over 5-year rolling window
- Measure correlation between "trading below 20th-30th percentile" and subsequent forward returns
- Select the valuation metric with strongest positive alpha correlation
- This "best valuation measure" becomes the primary feature for that company

### Feature Engineering

**Company-Specific Historical Percentiles** (5-year rolling window):
- Current valuation vs own historical distribution
- Percentile rankings: Full distribution (0-100)
- Ratios: Current value / P25, P50, P75 historical thresholds

**Industry-Relative Metrics**:
- Current valuation vs industry peer distribution (same 5-year window)
- Industry quantile ratios: Current value / Industry P25, P50, P75
- Custom industry taxonomy with 25+ sectors

**Winsorization**:
- All continuous features winsorized at 5th/95th percentiles within quarter
- Prevents extreme outlier distortion
- Applied before percentile calculations and model training

**Alpha Correlation Features**:
- Best valuation measure identifier per company
- Alpha correlation strength score
- Dynamic weighting based on correlation stability

### XGBoost Model Configuration

**Model Type**: Boosted Tree Classifier (Binary Classification)

**Target Variable**: 
- `investment_grade` (Boolean)
- Based on forward return thresholds and quality criteria
- Splits stocks into investment-grade vs non-investment-grade

**Hyperparameters** (consistent across regime models):
```yaml
model_type: BOOSTED_TREE_CLASSIFIER
num_parallel_tree: 6
max_tree_depth: 5
max_iterations: 400
learn_rate: 0.03
subsample: 0.7
l1_reg: 1.0
l2_reg: 0.5
min_rel_progress: 0.0001
data_split_method: CUSTOM
```

**Model Structure**:
- **4 regime-specific models per fold** (k1, k2, k3, k4)
- Separate classifier and regressor variants
- Each model trained on expanding window of historical data
- Out-of-time validation on future periods

**Training Approach**:
- Forward-step validation: K1 trains on early data, validates on later; K2 expands window, etc.
- Custom data splits based on temporal k-fold assignments
- Minimum training window ensures sufficient data per regime
- Monthly/quarterly rebalancing

## Feature Engineering

### Data Preprocessing

**Outlier Treatment**:
- Winsorization at 5th and 95th percentiles (applied within each quarter)
- Missing value handling through sector median imputation
- Time-series gap filling with forward fill (limited to 1 quarter max)

**Normalization**:
- Historical percentile ranking within company (5-year window)
- Historical percentile ranking within industry (5-year window)
- Standard scaling for macroeconomic PCA inputs
- Z-score normalization optional for certain features

**Lag Structure**:
- Fundamental data: 1-quarter lag minimum (reporting delays)
- Market data: Point-in-time snapshots at quarter end
- Macro data: Aligned to quarter-end
- Economic regime: 1-quarter lag to avoid look-ahead

### Critical Features

Based on the actual implementation, these are the key feature categories (features marked as EXCLUDED are not used):

**Valuation Features** (Primary):
- Adaptive "best valuation measure" per company
- Company-specific percentile rankings
- Industry-relative ratios (vs P25, P50, P75)
- Alpha correlation measures

**Quality Features**:
- ROE-related metrics and trends
- Profitability persistence measures
- Debt metrics and balance sheet strength
- Cash flow quality indicators

**Growth Features**:
- 10-year CAGRs (revenue, EPS, total equity)
- Quarter-over-quarter trends
- Industry growth relative positioning

**Features Explicitly EXCLUDED** (from the code):
- Raw price levels
- Market cap (used only for filtering)
- Raw volatility metrics (macro and company)
- Certain momentum indicators (3mo, 12mo price momentum excluded)
- Raw valuation multiples (only percentile versions used)
- Dividend yield
- Several macro variables (used only in regime classification, not stock model)

### Feature Validation

**Correlation Management**: 
- Multicollinearity addressed through feature exclusion
- Many correlated features removed (see EXCLUDE list in code)
- Focus on orthogonal information sources

**Stability Requirements**: 
- Features must have 5 years of history for percentile calculations
- Companies without sufficient history excluded from training
- Forward-fill limited to 1 quarter to prevent stale data

## Risk Management

### Model Risk Controls

**Overfitting Prevention**:
- Forward-step validation (no data leakage)
- Custom temporal data splits
- Out-of-time testing on completely holdout periods
- Separate validation metrics tracked per fold and regime
- L1/L2 regularization (1.0 and 0.5 respectively)

**Data Quality**:
- Winsorization limits extreme value impact
- Missing data imputation using sector medians
- Filters: price > 0, next_year_price > 0, market_cap > 0
- Minimum history requirements (5 years for percentiles)

**Position Limits** (implementation-dependent):
- Binary classification output enables flexible position sizing
- Probability scores allow ranking within investment-grade universe
- Recommended: Max single stock 3%, max sector 25%, min market cap $2B

### Model Monitoring

**Performance Tracking**:
- ROC-AUC tracked per fold and regime
- Precision, recall, F1-score calculated on validation sets
- Feature importance monitored via ML.FEATURE_IMPORTANCE
- Out-of-time performance compared to in-sample

**Regime Transition Handling**:
- Quarterly regime updates with 1-quarter lag
- Potential for regime smoothing (not currently implemented)
- Model selection based on predicted regime

**Retraining Schedule**:
- Quarterly retraining recommended (aligns with fundamental data)
- Annual full model rebuild (including regime classifier)
- Continuous monitoring of feature drift and model decay

### Known Limitations

1. **Regime Classification**:
   - Uses only 3 clusters (may miss nuance in some market conditions)
   - PCA trained on 1985-2005 period may not capture modern market dynamics
   - Quarterly updates create lag in regime adaptation

2. **Valuation Metric Selection**:
   - Requires 5-year history per company (excludes recent IPOs)
   - Alpha correlation based on historical data (may not predict future best metric)
   - Fixed 20th-30th percentile thresholds may not be optimal for all stocks

3. **Binary Classification**:
   - Investment-grade threshold is static
   - Loss of granularity vs regression approach
   - May underweight probability scores if using simple binary output

4. **Data Dependencies**:
   - Heavily dependent on clean fundamental data
   - Industry classification quality affects percentile calculations
   - Macro data lag in FRED sources

## Contributing

Contributions welcome! Areas for improvement:

1. **Regime Enhancement**: 
   - Experiment with 4-5 clusters
   - Update PCA training period to include recent decades
   - Add regime transition smoothing logic

2. **Feature Engineering**:
   - Expand to more quality metrics
   - Add analyst sentiment features
   - Incorporate options-based risk measures

3. **Model Architecture**:
   - Test ensemble approaches across regime models
   - Implement stacking with regime probabilities
   - Add calibration layer for probability scores

4. **Validation**:
   - Expand out-of-sample testing periods
   - Add transaction cost simulation
   - Implement Monte Carlo sensitivity analysis

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/value-quality-stock-selection.git

# Set up BigQuery access
gcloud auth application-default login

# Configure your project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Load sample data and test queries
bq query --use_legacy_sql=false < test_queries/validate_setup.sql
```

## Disclaimer

**Important Risk Disclosure**: This model is for educational and research purposes only. All backtests shown are simulated and may not reflect actual trading results. The model may experience significant losses during regime transitions or periods not well-represented in training data.

**Not Investment Advice**: This software is not intended as investment advice or recommendations. Past performance does not indicate future results. Users should conduct their own due diligence and consult with qualified financial advisors.

**Model Limitations**:
- **Data Quality Dependency**: Model performance critically depends on clean, accurate fundamental and macroeconomic data
- **Regime Detection Lag**: Economic regime classification operates on quarterly lag, may miss rapid market transitions
- **Historical Bias**: Valuation metric selection based on historical alpha may not persist in future periods
- **Survivorship Bias**: Ensure training data includes delisted/bankrupt companies to avoid overstating performance
- **Transaction Costs**: Backtest results do not reflect trading costs, market impact, or implementation shortfall
- **Overfitting Risk**: Despite validation procedures, in-sample optimization may not generalize to future markets

**Technical Limitations**:
- Requires substantial BigQuery compute resources
- 5-year minimum history requirement excludes many stocks
- Binary classification loses information vs probabilistic ranking
- Fixed industry taxonomy may become stale over time

## License

MIT License - See LICENSE file for details.

**Academic Use**: Please cite this work in academic research:
```
@software{value_quality_model_2025,
  title={Adaptive Valuation Stock Selection Model with PCA-Based Regime Classification},
  author={Matt Imig},
  year={2025},
  url={https://github.com/your-username/value-quality-stock-selection}
}
```

---

**Model Version**: 2.1.0  
**Last Updated**: November 2024  
**BigQuery SQL Version**: Standard SQL 2011  
**Implementation**: Pure BigQuery SQL (no Python required for core model)

For questions, issues, or collaboration opportunities, please open a GitHub issue or contact [matthewimig89@gmail.com].
