# Big Data Cruncher: Marketplace Fraud Detection Engine

A comprehensive statistical anomaly detection and pattern mining engine for European secondhand marketplace analysis.

## Overview

This project analyzes 10,000+ synthetic marketplace listings across multiple platforms (Vinted, eBay.de, Mercari EU, Facebook Marketplace, OLX) to detect fraud, uncover patterns, and extract actionable insights from massive real-world-style datasets.

## Features

- **7-Phase Analysis Pipeline**: From data generation to pattern discovery
- **Multi-Layer Fraud Detection**: 5-layer anomaly detection system
- **Advanced Statistical Analysis**: Distribution fitting, outlier detection, correlation analysis
- **Machine Learning**: K-Means, hierarchical clustering, DBSCAN, Isolation Forest
- **Time Series Analysis**: Seasonal patterns, price evolution, temporal trends
- **Association Rule Mining**: Market basket analysis to discover hidden patterns

## Dataset

**Size**: 10,000 records
**Timeframe**: 12 months
**Categories**: Clothing, Electronics, Furniture, Books, Collectibles
**Platforms**: Vinted, eBay.de, Mercari EU, Facebook Marketplace, OLX

### Data Features (24 columns):
- Item attributes: category, brand, condition, age_years, original_price, current_price
- Listing details: days_listed, description_length, image_count, shipping_offered
- Engagement metrics: view_count, favorite_count
- Seller information: seller_rating, seller_account_age_days, seller_item_count
- Behavioral data: price_drops, relistings
- Outcomes: sold, sale_price, days_to_sell
- Fraud indicators: suspected_counterfeit, fraud_flag

## Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for package dependencies

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis
```bash
python run_complete_analysis.py
```

This executes all 7 phases sequentially and generates comprehensive reports.

### Run Individual Phases
```bash
# Phase 1: Generate dataset
python phase1_data_generation.py

# Phase 2: Univariate analysis
python phase2_univariate_analysis.py

# Phase 3: Correlation analysis
python phase3_multivariate_analysis.py

# Phase 4: Categorical analysis
python phase4_categorical_analysis.py

# Phase 5: Anomaly detection
python phase5_anomaly_detection.py

# Phase 6: Pattern discovery
python phase6_pattern_discovery.py

# Phase 7: Time series analysis
python phase7_timeseries_analysis.py
```

## Analysis Pipeline

### Phase 1: Data Generation & Profiling
- Generates 10,000 realistic marketplace listings
- Implements correlations (brand value → price retention)
- Simulates seasonality (winter goods in summer slump)
- Injects anomalies and fraud patterns
- Outputs: `marketplace_data.csv`, `correlation_matrix.csv`

### Phase 2: Univariate Analysis
- Exhaustive statistical analysis per numeric column
- Distribution fitting (Normal, Log-Normal, Exponential, Gamma)
- Multi-method outlier detection (IQR, Z-score, Modified Z-score)
- Extreme value analysis
- Outputs: `univariate_analysis_results.json`

### Phase 3: Multivariate Correlation
- Full Pearson correlation matrix
- Strong/moderate/weak correlation identification
- Non-linear relationship exploration
- Causal inference analysis
- Outputs: `multivariate_correlation_matrix.csv`

### Phase 4: Categorical & Segmented Analysis
- Analysis by category (5 categories)
- Analysis by platform (5 platforms)
- Analysis by seller experience (4 tiers)
- Analysis by condition (5 conditions)
- Identifies category-specific patterns and premiums

### Phase 5: Anomaly & Fraud Detection
**5-Layer Detection System:**

1. **Price Anomalies**: Extreme low/high prices, rapid drops
2. **Behavioral Anomalies**: High views with no favorites, new bulk sellers
3. **Content Anomalies**: Missing images, minimal descriptions
4. **Luxury Counterfeits**: Low-priced luxury goods, suspicious seller patterns
5. **Statistical Outliers**: Isolation Forest, Mahalanobis distance

**Outputs**:
- `anomaly_detection_report.csv` (top 100 anomalies)
- `anomaly_scores.csv` (scores for all items)
- Fraud confidence scores (0-100%)

### Phase 6: Pattern Discovery
- **K-Means Clustering**: 5-8 natural seller/buyer profiles
- **Hierarchical Clustering**: Dendrogram analysis, super-clusters
- **DBSCAN**: Density-based clustering, outlier identification
- **Association Rules**: Market basket analysis, conditional patterns

**Outputs**:
- `cluster_analysis.csv`
- `association_rules.csv`

### Phase 7: Time Series Analysis
- Seasonal patterns by category
- Holiday effects (pre-holiday peaks, post-holiday liquidation)
- Price evolution and depreciation rates
- Platform trends
- Seller learning curves

## Key Insights Generated

### Fraud Detection
- Luxury items priced 50%+ below market = 99% counterfeit probability
- New sellers (<30 days) with bulk luxury listings = extreme fraud risk
- High views but zero favorites = misleading listings

### Pricing Strategies
- Electronics depreciate 3-5% per month
- Collectibles can appreciate with age
- Optimal price drops: 1-2 corrections after 30-60 days
- 3+ price drops signal desperation/quality issues

### Platform Recommendations
- Best sale rate platform identified
- Lowest fraud platform identified
- Highest average price platform identified

### Seller Insights
- Experienced sellers (100+ items) have higher sale rates
- Description length sweet spot: 300-500 characters
- 3-5 images optimal for engagement
- New sellers learn and improve metrics over time

## Output Files

| File | Description |
|------|-------------|
| `marketplace_data.csv` | 10,000 marketplace listings |
| `correlation_matrix.csv` | Pearson correlations |
| `univariate_analysis_results.json` | Statistical analysis per column |
| `multivariate_correlation_matrix.csv` | Full correlation matrix |
| `anomaly_detection_report.csv` | Top 100 anomalous items |
| `anomaly_scores.csv` | Anomaly scores for all items |
| `cluster_analysis.csv` | Cluster profiles and metrics |
| `association_rules.csv` | Discovered association rules |

## Statistical Methods Used

### Distributions
- Normal, Log-Normal, Exponential, Gamma distribution fitting
- Kolmogorov-Smirnov goodness-of-fit tests
- Anderson-Darling tests

### Outlier Detection
- IQR Method (1.5 × IQR)
- Z-Score (|z| > 3)
- Modified Z-Score (MAD-based)
- Isolation Forest
- Mahalanobis Distance

### Clustering
- K-Means with silhouette analysis
- Agglomerative hierarchical clustering
- DBSCAN (density-based)

### Correlation
- Pearson correlation
- Spearman rank correlation (non-linear relationships)

## Performance

- **Dataset Size**: 10,000 records
- **Estimated Runtime**: ~2-5 minutes (full pipeline)
- **Memory Usage**: ~500 MB peak

## Future Enhancements

- [ ] Visualization dashboard (matplotlib/plotly)
- [ ] Real-time fraud detection API
- [ ] Deep learning models for image authenticity
- [ ] NLP for description analysis
- [ ] Geographic fraud pattern analysis
- [ ] Cross-platform price arbitrage detection

## License

This is a demonstration project for statistical analysis and fraud detection methodologies.

## Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: All data is synthetically generated for demonstration purposes. No real marketplace data is used.
