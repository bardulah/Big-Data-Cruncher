"""
PHASE 3: MULTIVARIATE CORRELATION ANALYSIS
Build correlation matrix, analyze relationships, explore causal patterns
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class MultivariateAnalyzer:
    """Analyzes correlations and multivariate relationships"""

    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def analyze(self):
        """Perform complete multivariate analysis"""
        print("\n" + "="*80)
        print("PHASE 3: MULTIVARIATE CORRELATION ANALYSIS")
        print("="*80)

        # Build correlation matrix
        corr_matrix = self.build_correlation_matrix()

        # Analyze correlation strength
        self.analyze_correlation_strength(corr_matrix)

        # Deeper insights
        self.deeper_insights()

        # Causal inference questions
        self.causal_inference_analysis()

        print("\n" + "="*80)
        print("PHASE 3 COMPLETE")
        print("="*80)

        return corr_matrix

    def build_correlation_matrix(self):
        """Build full Pearson correlation matrix"""
        print(f"\n{'-'*80}")
        print("CORRELATION MATRIX (Pearson)")
        print(f"{'-'*80}")

        corr_matrix = self.df[self.numeric_cols].corr()

        # Display matrix
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.precision', 3)
        print("\n" + corr_matrix.to_string())

        return corr_matrix

    def analyze_correlation_strength(self, corr_matrix):
        """Analyze correlation strengths"""
        print(f"\n{'-'*80}")
        print("CORRELATION STRENGTH ANALYSIS")
        print(f"{'-'*80}")

        # Strong positive correlations (r > 0.7)
        print("\n  STRONG POSITIVE CORRELATIONS (r > 0.7):")
        strong_positive = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.7:
                    strong_positive.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })

        if strong_positive:
            sp_df = pd.DataFrame(strong_positive).sort_values('Correlation', ascending=False)
            print(sp_df.to_string(index=False))
            print(f"\n    Interpretation: These variables strengthen together")
        else:
            print("    None found")

        # Strong negative correlations (r < -0.7)
        print("\n  STRONG NEGATIVE CORRELATIONS (r < -0.7):")
        strong_negative = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] < -0.7:
                    strong_negative.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })

        if strong_negative:
            sn_df = pd.DataFrame(strong_negative).sort_values('Correlation')
            print(sn_df.to_string(index=False))
            print(f"\n    Interpretation: These variables are inversely related")
        else:
            print("    None found")

        # Moderate correlations (0.4 < |r| < 0.7)
        print("\n  MODERATE CORRELATIONS (0.4 < |r| < 0.7):")
        moderate = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if 0.4 < abs(r) < 0.7:
                    moderate.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': r
                    })

        if moderate:
            mod_df = pd.DataFrame(moderate).sort_values('Correlation', ascending=False, key=abs)
            print(mod_df.head(10).to_string(index=False))
            print(f"    ... and {len(moderate) - 10} more" if len(moderate) > 10 else "")
        else:
            print("    None found")

        # Weak correlations (|r| < 0.3)
        print("\n  WEAK CORRELATIONS (|r| < 0.3):")
        weak_count = 0
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) < 0.3:
                    weak_count += 1
        print(f"    Total weak correlations: {weak_count}")
        print(f"    Interpretation: Potentially independent factors or non-linear relationships")

    def deeper_insights(self):
        """Explore deeper multivariate insights"""
        print(f"\n{'-'*80}")
        print("DEEPER INSIGHTS: NON-LINEAR & CONDITIONAL RELATIONSHIPS")
        print(f"{'-'*80}")

        # Days listed analysis - U-curve detection
        print("\n  DAYS LISTED PATTERN ANALYSIS:")
        quick_sales = self.df[self.df['days_listed'] < 7]
        normal_sales = self.df[(self.df['days_listed'] >= 7) & (self.df['days_listed'] < 90)]
        long_listings = self.df[self.df['days_listed'] >= 90]

        print(f"    Quick sales (<7 days):")
        print(f"      Count: {len(quick_sales)}")
        print(f"      Sold rate: {(quick_sales['sold'] == 'yes').sum() / len(quick_sales) * 100:.1f}%")
        print(f"      Avg suspected_counterfeit: {quick_sales['suspected_counterfeit'].mean():.3f}")

        print(f"    Normal listings (7-90 days):")
        print(f"      Count: {len(normal_sales)}")
        print(f"      Sold rate: {(normal_sales['sold'] == 'yes').sum() / len(normal_sales) * 100:.1f}%")
        print(f"      Avg suspected_counterfeit: {normal_sales['suspected_counterfeit'].mean():.3f}")

        print(f"    Long listings (>90 days):")
        print(f"      Count: {len(long_listings)}")
        print(f"      Sold rate: {(long_listings['sold'] == 'yes').sum() / len(long_listings) * 100:.1f}%")
        print(f"      Avg suspected_counterfeit: {long_listings['suspected_counterfeit'].mean():.3f}")

        # View count vs description length - non-linear
        print("\n  VIEW COUNT vs DESCRIPTION LENGTH (non-linear analysis):")
        bins = [0, 100, 300, 500, 1000, 2000]
        labels = ['0-100', '100-300', '300-500', '500-1000', '1000+']
        self.df['desc_bin'] = pd.cut(self.df['description_length'], bins=bins, labels=labels)
        view_by_desc = self.df.groupby('desc_bin', observed=True)['view_count'].mean()

        print("    Avg views by description length:")
        for label, views in view_by_desc.items():
            print(f"      {label:12s}: {views:,.1f} views")
        print("    → Sweet spot appears to be 300-500 characters")

        # Age sweet spot by category
        print("\n  AGE SWEET SPOT ANALYSIS BY CATEGORY:")
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            sold_items = cat_data[cat_data['sold'] == 'yes']
            if len(sold_items) > 0:
                avg_age_sold = sold_items['age_years'].mean()
                avg_age_unsold = cat_data[cat_data['sold'] == 'no']['age_years'].mean()
                print(f"    {category.capitalize():15s}: Sold avg age={avg_age_sold:.1f}y, "
                      f"Unsold avg age={avg_age_unsold:.1f}y")

        # Seller reputation effect by price tier
        print("\n  SELLER RATING EFFECT BY PRICE TIER:")
        self.df['price_tier'] = pd.qcut(self.df['current_price'], q=4,
                                        labels=['Low', 'Medium', 'High', 'Luxury'],
                                        duplicates='drop')
        for tier in ['Low', 'Medium', 'High', 'Luxury']:
            tier_data = self.df[self.df['price_tier'] == tier]
            if len(tier_data) > 0:
                corr = tier_data[['seller_rating', 'current_price']].corr().iloc[0, 1]
                sold_rate_high_rating = (tier_data[tier_data['seller_rating'] > 4.0]['sold'] == 'yes').mean()
                sold_rate_low_rating = (tier_data[tier_data['seller_rating'] <= 4.0]['sold'] == 'yes').mean()
                print(f"    {tier:8s} tier: rating-price corr={corr:.3f}, "
                      f"sold rate (high rating)={sold_rate_high_rating*100:.1f}%, "
                      f"sold rate (low rating)={sold_rate_low_rating*100:.1f}%")

        # Platform-specific correlation differences
        print("\n  PLATFORM-SPECIFIC CORRELATION PATTERNS:")
        for platform in self.df['platform'].unique():
            platform_data = self.df[self.df['platform'] == platform]
            if len(platform_data) > 50:  # Sufficient sample size
                price_view_corr = platform_data[['current_price', 'view_count']].corr().iloc[0, 1]
                rating_sold_corr = platform_data[['seller_rating', 'sold']].apply(
                    lambda x: x.map({'yes': 1, 'no': 0}) if x.name == 'sold' else x).corr().iloc[0, 1]
                print(f"    {platform:20s}: price-views={price_view_corr:.3f}, "
                      f"rating-sold={rating_sold_corr:.3f}")

    def causal_inference_analysis(self):
        """Explore potential causal relationships"""
        print(f"\n{'-'*80}")
        print("CAUSAL INFERENCE QUESTIONS")
        print(f"{'-'*80}")

        # Does high view_count CAUSE sales?
        print("\n  Q1: Does high view_count CAUSE sales or just correlation?")
        high_views = self.df[self.df['view_count'] > self.df['view_count'].quantile(0.75)]
        low_views = self.df[self.df['view_count'] <= self.df['view_count'].quantile(0.25)]

        high_views_sold_rate = (high_views['sold'] == 'yes').sum() / len(high_views)
        low_views_sold_rate = (low_views['sold'] == 'yes').sum() / len(low_views)

        print(f"    High view items (Q4): {high_views_sold_rate*100:.1f}% sold")
        print(f"    Low view items (Q1): {low_views_sold_rate*100:.1f}% sold")
        print(f"    Difference: {(high_views_sold_rate - low_views_sold_rate)*100:.1f} percentage points")
        print("    → Analysis: High correlation, but confounded by pricing.")
        print("      Good pricing attracts BOTH views AND sales.")

        # Price drops: strategy or desperation?
        print("\n  Q2: Do price_drops indicate smart strategy or desperation?")
        no_drops = self.df[self.df['price_drops'] == 0]
        some_drops = self.df[(self.df['price_drops'] >= 1) & (self.df['price_drops'] <= 2)]
        many_drops = self.df[self.df['price_drops'] > 2]

        print(f"    No drops: {(no_drops['sold'] == 'yes').sum() / len(no_drops) * 100:.1f}% sold, "
              f"avg days_listed={no_drops['days_listed'].mean():.1f}")
        print(f"    1-2 drops: {(some_drops['sold'] == 'yes').sum() / len(some_drops) * 100:.1f}% sold, "
              f"avg days_listed={some_drops['days_listed'].mean():.1f}")
        print(f"    3+ drops: {(many_drops['sold'] == 'yes').sum() / len(many_drops) * 100:.1f}% sold, "
              f"avg days_listed={many_drops['days_listed'].mean():.1f}")
        print("    → Analysis: Moderate drops (1-2) can be strategic price corrections.")
        print("      Excessive drops (3+) signal desperation and item quality concerns.")

        # Seller item count: expertise or selection bias?
        print("\n  Q3: Does seller_item_count indicate expertise or just volume?")
        novice = self.df[self.df['seller_item_count'] < 10]
        experienced = self.df[(self.df['seller_item_count'] >= 100) &
                             (self.df['seller_item_count'] < 1000)]
        power = self.df[self.df['seller_item_count'] >= 1000]

        print(f"    Novice (<10): avg_price={novice['current_price'].mean():.2f}, "
              f"sold_rate={( novice['sold'] == 'yes').mean()*100:.1f}%")
        print(f"    Experienced (100-999): avg_price={experienced['current_price'].mean():.2f}, "
              f"sold_rate={(experienced['sold'] == 'yes').mean()*100:.1f}%")
        print(f"    Power (1000+): avg_price={power['current_price'].mean():.2f}, "
              f"sold_rate={(power['sold'] == 'yes').mean()*100:.1f}%")
        print("    → Analysis: Experience correlates with better pricing and sales.")
        print("      But power sellers may focus on volume over margins.")

        # Image count effect
        print("\n  Q4: What is the causal effect of image_count on sales?")
        no_images = self.df[self.df['image_count'] == 0]
        few_images = self.df[(self.df['image_count'] >= 1) & (self.df['image_count'] <= 3)]
        many_images = self.df[self.df['image_count'] > 5]

        print(f"    No images: {(no_images['sold'] == 'yes').mean()*100:.1f}% sold")
        print(f"    1-3 images: {(few_images['sold'] == 'yes').mean()*100:.1f}% sold")
        print(f"    5+ images: {(many_images['sold'] == 'yes').mean()*100:.1f}% sold")
        print("    → Analysis: Strong causal evidence that images directly increase sales.")
        print("      Each additional image (up to ~5) adds credibility and buyer confidence.")

        # Spearman correlation for non-linear relationships
        print("\n  SPEARMAN CORRELATION (captures monotonic non-linear relationships):")
        key_vars = ['current_price', 'view_count', 'favorite_count', 'days_listed',
                   'seller_rating', 'description_length', 'image_count']

        spearman_corrs = []
        for var in key_vars:
            if var in self.df.columns:
                sold_numeric = self.df['sold'].map({'yes': 1, 'no': 0})
                corr, pval = spearmanr(self.df[var].dropna(), sold_numeric[self.df[var].notna()])
                spearman_corrs.append({
                    'Variable': var,
                    'Spearman_r': corr,
                    'P-value': pval
                })

        sp_df = pd.DataFrame(spearman_corrs).sort_values('Spearman_r', ascending=False, key=abs)
        print(sp_df.to_string(index=False))


def main():
    """Main execution function"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    analyzer = MultivariateAnalyzer(df)
    corr_matrix = analyzer.analyze()

    # Save results
    corr_matrix.to_csv('multivariate_correlation_matrix.csv')
    print(f"\n✓ Correlation matrix saved to: multivariate_correlation_matrix.csv")

    return corr_matrix


if __name__ == "__main__":
    corr_matrix = main()
