"""
PHASE 2: UNIVARIATE ANALYSIS DEEP DIVE
Exhaustive statistical analysis of each numeric variable
Includes distribution fitting, outlier detection, and extreme value analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (norm, lognorm, poisson, expon, gamma,
                         kstest, anderson, shapiro)
import warnings
warnings.filterwarnings('ignore')


class UnivariateAnalyzer:
    """Performs comprehensive univariate statistical analysis"""

    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.results = {}

    def analyze_all_columns(self):
        """Perform comprehensive analysis on all numeric columns"""
        print("\n" + "="*80)
        print("PHASE 2: UNIVARIATE ANALYSIS DEEP DIVE")
        print("="*80)

        for col in self.numeric_cols:
            print(f"\n{'='*80}")
            print(f"ANALYZING: {col.upper()}")
            print(f"{'='*80}")

            result = self.analyze_column(col)
            self.results[col] = result

        # Summary report
        self.generate_summary_report()

        return self.results

    def analyze_column(self, col):
        """Comprehensive analysis of a single column"""
        data = self.df[col].dropna()

        if len(data) == 0:
            print(f"  [WARNING] Column '{col}' has no valid data")
            return None

        result = {
            'column': col,
            'descriptive': self.descriptive_statistics(data, col),
            'distribution': self.distribution_analysis(data, col),
            'outliers': self.outlier_detection(data, col),
            'extremes': self.extreme_value_analysis(data, col)
        }

        return result

    def descriptive_statistics(self, data, col):
        """Calculate comprehensive descriptive statistics"""
        print(f"\n{'-'*80}")
        print(f"DESCRIPTIVE STATISTICS: {col}")
        print(f"{'-'*80}")

        stats_dict = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode()[0] if len(data.mode()) > 0 else np.nan,
            'std': data.std(),
            'variance': data.var(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }

        # Quantiles
        quantiles = {
            'p01': data.quantile(0.01),
            'p05': data.quantile(0.05),
            'p25': data.quantile(0.25),
            'p50': data.quantile(0.50),
            'p75': data.quantile(0.75),
            'p95': data.quantile(0.95),
            'p99': data.quantile(0.99)
        }

        print(f"  Count:               {stats_dict['count']:,.0f}")
        print(f"  Mean:                {stats_dict['mean']:,.2f}")
        print(f"  Median:              {stats_dict['median']:,.2f}")
        print(f"  Mode:                {stats_dict['mode']:,.2f}")
        print(f"  Std Dev:             {stats_dict['std']:,.2f}")
        print(f"  Variance:            {stats_dict['variance']:,.2f}")
        print(f"  CV:                  {stats_dict['cv']:.3f}")
        print(f"\n  Range:               {stats_dict['range']:,.2f}")
        print(f"  Min:                 {stats_dict['min']:,.2f}")
        print(f"  Max:                 {stats_dict['max']:,.2f}")
        print(f"\n  Quantiles:")
        for q, val in quantiles.items():
            print(f"    {q.upper():8s}             {val:,.2f}")
        print(f"\n  IQR:                 {stats_dict['iqr']:,.2f}")
        print(f"  Skewness:            {stats_dict['skewness']:.4f}", end="")

        # Interpret skewness
        if abs(stats_dict['skewness']) < 0.5:
            print(" (Approximately symmetric)")
        elif stats_dict['skewness'] > 0.5:
            print(" (Right-skewed / Positive skew)")
        else:
            print(" (Left-skewed / Negative skew)")

        print(f"  Kurtosis:            {stats_dict['kurtosis']:.4f}", end="")

        # Interpret kurtosis
        if abs(stats_dict['kurtosis']) < 0.5:
            print(" (Mesokurtic - normal tails)")
        elif stats_dict['kurtosis'] > 0.5:
            print(" (Leptokurtic - heavy tails)")
        else:
            print(" (Platykurtic - light tails)")

        stats_dict['quantiles'] = quantiles
        return stats_dict

    def distribution_analysis(self, data, col):
        """Fit multiple distributions and find best fit"""
        print(f"\n{'-'*80}")
        print(f"DISTRIBUTION ANALYSIS: {col}")
        print(f"{'-'*80}")

        # Ensure positive data for certain distributions
        data_positive = data[data > 0] if (data <= 0).any() else data

        fits = {}

        # Normal distribution
        try:
            mu, sigma = norm.fit(data)
            ks_stat, ks_pval = kstest(data, 'norm', args=(mu, sigma))
            fits['normal'] = {
                'params': {'mu': mu, 'sigma': sigma},
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval
            }
            print(f"\n  Normal Distribution:")
            print(f"    Parameters: μ={mu:.2f}, σ={sigma:.2f}")
            print(f"    KS Statistic: {ks_stat:.4f}")
            print(f"    KS p-value: {ks_pval:.4f}")
            print(f"    Fit Quality: {'GOOD' if ks_pval > 0.05 else 'POOR'}")
        except Exception as e:
            fits['normal'] = None
            print(f"  Normal Distribution: Failed to fit ({str(e)})")

        # Log-Normal distribution (only for positive data)
        if len(data_positive) > 0:
            try:
                shape, loc, scale = lognorm.fit(data_positive)
                ks_stat, ks_pval = kstest(data_positive, 'lognorm', args=(shape, loc, scale))
                fits['lognormal'] = {
                    'params': {'shape': shape, 'loc': loc, 'scale': scale},
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval
                }
                print(f"\n  Log-Normal Distribution:")
                print(f"    Parameters: shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f}")
                print(f"    KS Statistic: {ks_stat:.4f}")
                print(f"    KS p-value: {ks_pval:.4f}")
                print(f"    Fit Quality: {'GOOD' if ks_pval > 0.05 else 'POOR'}")
            except Exception as e:
                fits['lognormal'] = None
                print(f"  Log-Normal Distribution: Failed to fit ({str(e)})")

        # Exponential distribution (only for positive data)
        if len(data_positive) > 0:
            try:
                loc, scale = expon.fit(data_positive)
                ks_stat, ks_pval = kstest(data_positive, 'expon', args=(loc, scale))
                fits['exponential'] = {
                    'params': {'loc': loc, 'scale': scale},
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval
                }
                print(f"\n  Exponential Distribution:")
                print(f"    Parameters: loc={loc:.2f}, scale={scale:.2f}")
                print(f"    KS Statistic: {ks_stat:.4f}")
                print(f"    KS p-value: {ks_pval:.4f}")
                print(f"    Fit Quality: {'GOOD' if ks_pval > 0.05 else 'POOR'}")
            except Exception as e:
                fits['exponential'] = None
                print(f"  Exponential Distribution: Failed to fit ({str(e)})")

        # Gamma distribution
        if len(data_positive) > 0:
            try:
                a, loc, scale = gamma.fit(data_positive)
                ks_stat, ks_pval = kstest(data_positive, 'gamma', args=(a, loc, scale))
                fits['gamma'] = {
                    'params': {'a': a, 'loc': loc, 'scale': scale},
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval
                }
                print(f"\n  Gamma Distribution:")
                print(f"    Parameters: a={a:.2f}, loc={loc:.2f}, scale={scale:.2f}")
                print(f"    KS Statistic: {ks_stat:.4f}")
                print(f"    KS p-value: {ks_pval:.4f}")
                print(f"    Fit Quality: {'GOOD' if ks_pval > 0.05 else 'POOR'}")
            except Exception as e:
                fits['gamma'] = None
                print(f"  Gamma Distribution: Failed to fit ({str(e)})")

        # Determine best fit
        best_fit = None
        best_pval = 0
        for dist_name, fit in fits.items():
            if fit is not None and fit['ks_pvalue'] > best_pval:
                best_pval = fit['ks_pvalue']
                best_fit = dist_name

        print(f"\n  BEST FIT: {best_fit.upper() if best_fit else 'NONE'}")
        if best_fit:
            print(f"    Reason: Highest KS p-value ({best_pval:.4f})")

        # Histogram binning
        n_bins = int(np.ceil(np.log2(len(data)) + 1))  # Sturges' rule
        print(f"\n  Histogram Analysis:")
        print(f"    Optimal bins (Sturges): {n_bins}")
        hist, bin_edges = np.histogram(data, bins=n_bins)
        print(f"    Most frequent bin: [{bin_edges[hist.argmax()]:.2f}, {bin_edges[hist.argmax()+1]:.2f}]")
        print(f"    Frequency: {hist.max()} ({hist.max()/len(data)*100:.1f}%)")

        return {
            'fits': fits,
            'best_fit': best_fit,
            'n_bins': n_bins
        }

    def outlier_detection(self, data, col):
        """Multi-method outlier detection"""
        print(f"\n{'-'*80}")
        print(f"OUTLIER DETECTION: {col}")
        print(f"{'-'*80}")

        outliers = {}

        # Method 1: IQR Method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        outliers['iqr'] = {
            'count': len(iqr_outliers),
            'indices': iqr_outliers.index.tolist(),
            'values': iqr_outliers.tolist(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        print(f"\n  Method 1: IQR Method (1.5 × IQR)")
        print(f"    Lower bound: {lower_bound:,.2f}")
        print(f"    Upper bound: {upper_bound:,.2f}")
        print(f"    Outliers found: {len(iqr_outliers)} ({len(iqr_outliers)/len(data)*100:.2f}%)")

        # Method 2: Z-Score (|z| > 3)
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]
        outliers['zscore'] = {
            'count': len(z_outliers),
            'indices': z_outliers.index.tolist(),
            'values': z_outliers.tolist(),
            'threshold': 3
        }
        print(f"\n  Method 2: Z-Score (|z| > 3)")
        print(f"    Outliers found: {len(z_outliers)} ({len(z_outliers)/len(data)*100:.2f}%)")

        # Method 3: Modified Z-Score (using MAD)
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros(len(data))
        mad_outliers = data[np.abs(modified_z_scores) > 3.5]
        outliers['modified_zscore'] = {
            'count': len(mad_outliers),
            'indices': mad_outliers.index.tolist(),
            'values': mad_outliers.tolist(),
            'threshold': 3.5,
            'mad': mad
        }
        print(f"\n  Method 3: Modified Z-Score (MAD-based, threshold=3.5)")
        print(f"    MAD: {mad:,.2f}")
        print(f"    Outliers found: {len(mad_outliers)} ({len(mad_outliers)/len(data)*100:.2f}%)")

        # Cross-validation: outliers identified by all methods
        if len(iqr_outliers) > 0 and len(z_outliers) > 0 and len(mad_outliers) > 0:
            consensus_outliers = set(iqr_outliers.index) & set(z_outliers.index) & set(mad_outliers.index)
            print(f"\n  Cross-Validation:")
            print(f"    Outliers agreed upon by ALL methods: {len(consensus_outliers)}")
            if len(consensus_outliers) > 0:
                print(f"    These are highly confident outliers")
                outliers['consensus'] = list(consensus_outliers)
        else:
            outliers['consensus'] = []

        return outliers

    def extreme_value_analysis(self, data, col):
        """Analyze extreme values"""
        print(f"\n{'-'*80}")
        print(f"EXTREME VALUE ANALYSIS: {col}")
        print(f"{'-'*80}")

        # Top 10 highest values
        top10 = data.nlargest(10)
        print(f"\n  TOP 10 HIGHEST VALUES:")
        for i, (idx, val) in enumerate(top10.items(), 1):
            print(f"    {i:2d}. Item {idx:5d}: {val:,.2f}")

        # Bottom 10 lowest values
        bottom10 = data.nsmallest(10)
        print(f"\n  BOTTOM 10 LOWEST VALUES:")
        for i, (idx, val) in enumerate(bottom10.items(), 1):
            print(f"    {i:2d}. Item {idx:5d}: {val:,.2f}")

        # Most anomalous (furthest from mean in terms of std devs)
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        most_anomalous_idx = z_scores.idxmax()
        most_anomalous_val = data[most_anomalous_idx]
        most_anomalous_z = z_scores[most_anomalous_idx]

        print(f"\n  MOST ANOMALOUS VALUE:")
        print(f"    Item {most_anomalous_idx}: {most_anomalous_val:,.2f}")
        print(f"    Z-score: {most_anomalous_z:.2f}")
        print(f"    Deviations from mean: {most_anomalous_z:.1f} standard deviations")

        # Plausibility assessment
        print(f"\n  PLAUSIBILITY ASSESSMENT:")
        extreme_threshold = mean + 5 * std
        extreme_count = (data > extreme_threshold).sum()
        print(f"    Values >5σ from mean: {extreme_count}")
        if extreme_count > 0:
            print(f"    → Possible data entry errors or genuine rarities")

        return {
            'top10': top10.to_dict(),
            'bottom10': bottom10.to_dict(),
            'most_anomalous': {
                'index': int(most_anomalous_idx),
                'value': float(most_anomalous_val),
                'zscore': float(most_anomalous_z)
            }
        }

    def generate_summary_report(self):
        """Generate summary report across all columns"""
        print("\n" + "="*80)
        print("SUMMARY REPORT: ALL NUMERIC VARIABLES")
        print("="*80)

        summary_data = []
        for col, result in self.results.items():
            if result is None:
                continue

            summary_data.append({
                'Column': col,
                'Mean': result['descriptive']['mean'],
                'Std': result['descriptive']['std'],
                'Skewness': result['descriptive']['skewness'],
                'Kurtosis': result['descriptive']['kurtosis'],
                'Best_Fit': result['distribution']['best_fit'] if result['distribution']['best_fit'] else 'None',
                'Outliers_IQR': result['outliers']['iqr']['count'],
                'Outliers_Z': result['outliers']['zscore']['count']
            })

        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

        print("\n" + "="*80)
        print("PHASE 2 COMPLETE")
        print("="*80)


def main():
    """Main execution function"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    analyzer = UnivariateAnalyzer(df)
    results = analyzer.analyze_all_columns()

    # Save results
    import json
    with open('univariate_analysis_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\n✓ Results saved to: univariate_analysis_results.json")

    return results


if __name__ == "__main__":
    results = main()
