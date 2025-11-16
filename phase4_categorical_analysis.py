"""
PHASE 4: CATEGORICAL & SEGMENTED ANALYSIS
Analyze patterns by category, platform, seller experience, and condition
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class CategoricalAnalyzer:
    """Analyzes categorical segments and groupings"""

    def __init__(self, df):
        self.df = df
        self.luxury_brands = ['Gucci', 'Louis Vuitton', 'Prada', 'Supreme',
                             'Apple', 'Herman Miller']

    def analyze(self):
        """Perform complete categorical analysis"""
        print("\n" + "="*80)
        print("PHASE 4: CATEGORICAL & SEGMENTED ANALYSIS")
        print("="*80)

        self.analyze_by_category()
        self.analyze_by_platform()
        self.analyze_by_seller_experience()
        self.analyze_by_condition()

        print("\n" + "="*80)
        print("PHASE 4 COMPLETE")
        print("="*80)

    def analyze_by_category(self):
        """Analyze patterns by product category"""
        print(f"\n{'-'*80}")
        print("ANALYSIS BY CATEGORY")
        print(f"{'-'*80}")

        for category in sorted(self.df['category'].unique()):
            cat_data = self.df[self.df['category'] == category]

            print(f"\n  {category.upper()}")
            print(f"  {'─' * 70}")

            # Basic statistics
            print(f"    Count: {len(cat_data):,} ({len(cat_data)/len(self.df)*100:.1f}%)")

            # Price statistics
            print(f"    Price (mean): €{cat_data['current_price'].mean():.2f}")
            print(f"    Price (median): €{cat_data['current_price'].median():.2f}")
            print(f"    Price (std): €{cat_data['current_price'].std():.2f}")
            price_cv = cat_data['current_price'].std() / cat_data['current_price'].mean()
            print(f"    Price volatility (CV): {price_cv:.3f}")

            # Sale success
            sold_rate = (cat_data['sold'] == 'yes').sum() / len(cat_data) * 100
            print(f"    Sale success rate: {sold_rate:.1f}%")

            # Listing duration
            sold_items = cat_data[cat_data['sold'] == 'yes']
            if len(sold_items) > 0:
                avg_days_to_sell = sold_items['days_to_sell'].mean()
                print(f"    Avg days to sell: {avg_days_to_sell:.1f}")
            else:
                print(f"    Avg days to sell: N/A")

            # Counterfeit risk
            counterfeit_rate = (cat_data['fraud_flag'] == True).sum() / len(cat_data) * 100
            avg_counterfeit_score = cat_data['suspected_counterfeit'].mean()
            print(f"    Fraud flags: {counterfeit_rate:.1f}%")
            print(f"    Avg counterfeit score: {avg_counterfeit_score:.3f}")

            # Seasonal pattern hint (simplified)
            if 'month_listed' in cat_data.columns:
                seasonal_high = cat_data.groupby('month_listed').size().idxmax()
                seasonal_low = cat_data.groupby('month_listed').size().idxmin()
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                print(f"    Peak listing month: {month_names[seasonal_high]}")
                print(f"    Low listing month: {month_names[seasonal_low]}")

            # Luxury brand premium
            if category in ['clothing', 'electronics']:
                luxury_items = cat_data[cat_data['brand'].isin(self.luxury_brands)]
                regular_items = cat_data[~cat_data['brand'].isin(self.luxury_brands)]
                if len(luxury_items) > 0 and len(regular_items) > 0:
                    premium = (luxury_items['current_price'].mean() /
                              regular_items['current_price'].mean())
                    print(f"    Luxury brand premium: {premium:.2f}x")

            # Price decay (age effect)
            if cat_data['age_years'].std() > 0:
                age_price_corr = cat_data[['age_years', 'current_price']].corr().iloc[0, 1]
                print(f"    Age-price correlation: {age_price_corr:.3f}")
                if age_price_corr < -0.3:
                    print(f"    → Strong depreciation with age")
                elif age_price_corr > 0.3:
                    print(f"    → Appreciation with age (vintage/collectible)")
                else:
                    print(f"    → Age has weak price impact")

    def analyze_by_platform(self):
        """Analyze patterns by platform"""
        print(f"\n{'-'*80}")
        print("ANALYSIS BY PLATFORM")
        print(f"{'-'*80}")

        platform_summary = []

        for platform in sorted(self.df['platform'].unique()):
            plat_data = self.df[self.df['platform'] == platform]

            print(f"\n  {platform.upper()}")
            print(f"  {'─' * 70}")

            # Market composition
            print(f"    Listings: {len(plat_data):,} ({len(plat_data)/len(self.df)*100:.1f}%)")
            print(f"    Category composition:")
            cat_dist = plat_data['category'].value_counts(normalize=True)
            for cat, pct in cat_dist.head(3).items():
                print(f"      {cat.capitalize():15s}: {pct*100:.1f}%")

            # Price levels
            avg_price = plat_data['current_price'].mean()
            print(f"    Avg price: €{avg_price:.2f}")

            # Seller demographics
            avg_seller_age = plat_data['seller_account_age_days'].mean()
            avg_seller_items = plat_data['seller_item_count'].mean()
            avg_seller_rating = plat_data['seller_rating'].mean()
            print(f"    Avg seller account age: {avg_seller_age:.0f} days ({avg_seller_age/365:.1f} years)")
            print(f"    Avg seller items: {avg_seller_items:.0f}")
            print(f"    Avg seller rating: {avg_seller_rating:.2f}/5.0")

            # Fraud rates
            fraud_rate = (plat_data['fraud_flag'] == True).sum() / len(plat_data) * 100
            print(f"    Fraud flag rate: {fraud_rate:.1f}%")

            # Time-to-sale efficiency
            sold_items = plat_data[plat_data['sold'] == 'yes']
            if len(sold_items) > 0:
                avg_days_to_sell = sold_items['days_to_sell'].mean()
                sale_rate = len(sold_items) / len(plat_data) * 100
                print(f"    Sale success rate: {sale_rate:.1f}%")
                print(f"    Avg days to sell: {avg_days_to_sell:.1f}")
            else:
                print(f"    Sale success rate: 0.0%")

            platform_summary.append({
                'Platform': platform,
                'Listings': len(plat_data),
                'Avg_Price': avg_price,
                'Sale_Rate_%': (plat_data['sold'] == 'yes').mean() * 100,
                'Fraud_Rate_%': fraud_rate,
                'Avg_Seller_Rating': avg_seller_rating
            })

        # Comparison table
        print(f"\n  PLATFORM COMPARISON TABLE:")
        print(f"  {'─' * 70}")
        summary_df = pd.DataFrame(platform_summary)
        print(summary_df.to_string(index=False))

        # Recommendations
        print(f"\n  RECOMMENDATIONS:")
        best_sale_rate = summary_df.loc[summary_df['Sale_Rate_%'].idxmax()]
        lowest_fraud = summary_df.loc[summary_df['Fraud_Rate_%'].idxmin()]
        highest_price = summary_df.loc[summary_df['Avg_Price'].idxmax()]

        print(f"    Best sale rate: {best_sale_rate['Platform']} ({best_sale_rate['Sale_Rate_%']:.1f}%)")
        print(f"    Lowest fraud: {lowest_fraud['Platform']} ({lowest_fraud['Fraud_Rate_%']:.1f}%)")
        print(f"    Highest avg price: {highest_price['Platform']} (€{highest_price['Avg_Price']:.2f})")

    def analyze_by_seller_experience(self):
        """Analyze patterns by seller experience level"""
        print(f"\n{'-'*80}")
        print("ANALYSIS BY SELLER EXPERIENCE")
        print(f"{'-'*80}")

        # Define experience tiers
        self.df['seller_tier'] = pd.cut(self.df['seller_item_count'],
                                        bins=[-1, 10, 100, 1000, 10000],
                                        labels=['Novice (<10)', 'Experienced (10-100)',
                                               'Power (100-1000)', 'Super (1000+)'])

        for tier in ['Novice (<10)', 'Experienced (10-100)', 'Power (100-1000)', 'Super (1000+)']:
            tier_data = self.df[self.df['seller_tier'] == tier]

            if len(tier_data) == 0:
                continue

            print(f"\n  {tier.upper()}")
            print(f"  {'─' * 70}")

            print(f"    Count: {len(tier_data):,} sellers")

            # Pricing
            avg_price = tier_data['current_price'].mean()
            print(f"    Avg listing price: €{avg_price:.2f}")

            # Sale success
            sale_rate = (tier_data['sold'] == 'yes').sum() / len(tier_data) * 100
            print(f"    Sale success rate: {sale_rate:.1f}%")

            # Fraud detection
            fraud_rate = (tier_data['fraud_flag'] == True).sum() / len(tier_data) * 100
            print(f"    Fraud flag rate: {fraud_rate:.1f}%")

            # Learning curve indicators
            avg_desc_length = tier_data['description_length'].mean()
            avg_images = tier_data['image_count'].mean()
            avg_price_drops = tier_data['price_drops'].mean()
            print(f"    Avg description length: {avg_desc_length:.0f} chars")
            print(f"    Avg images per listing: {avg_images:.1f}")
            print(f"    Avg price drops: {avg_price_drops:.2f}")

            # Time to sell
            sold_items = tier_data[tier_data['sold'] == 'yes']
            if len(sold_items) > 0:
                avg_days = sold_items['days_to_sell'].mean()
                print(f"    Avg days to sell: {avg_days:.1f}")

            # Price premium for reputation
            avg_rating = tier_data['seller_rating'].mean()
            print(f"    Avg seller rating: {avg_rating:.2f}/5.0")

        # Learning curve analysis
        print(f"\n  LEARNING CURVE INSIGHTS:")
        print(f"  {'─' * 70}")
        print("    As sellers gain experience:")

        novice = self.df[self.df['seller_tier'] == 'Novice (<10)']
        experienced = self.df[self.df['seller_tier'] == 'Experienced (10-100)']

        if len(novice) > 0 and len(experienced) > 0:
            desc_improvement = ((experienced['description_length'].mean() -
                               novice['description_length'].mean()) /
                              novice['description_length'].mean() * 100)
            sale_improvement = ((experienced['sold'] == 'yes').mean() -
                              (novice['sold'] == 'yes').mean()) * 100

            print(f"    Description length increases: {desc_improvement:+.1f}%")
            print(f"    Sale rate improves: {sale_improvement:+.1f} percentage points")
            print(f"    → Experienced sellers write better descriptions and sell more")

    def analyze_by_condition(self):
        """Analyze patterns by item condition"""
        print(f"\n{'-'*80}")
        print("ANALYSIS BY CONDITION")
        print(f"{'-'*80}")

        conditions = ['mint', 'like-new', 'good', 'fair', 'poor']

        condition_summary = []

        for condition in conditions:
            cond_data = self.df[self.df['condition'] == condition]

            if len(cond_data) == 0:
                continue

            print(f"\n  {condition.upper()}")
            print(f"  {'─' * 70}")

            # Count
            print(f"    Count: {len(cond_data):,} ({len(cond_data)/len(self.df)*100:.1f}%)")

            # Price multiplier (relative to 'good' condition)
            avg_price = cond_data['current_price'].mean()
            print(f"    Avg price: €{avg_price:.2f}")

            # Engagement
            avg_views = cond_data['view_count'].mean()
            avg_favs = cond_data['favorite_count'].mean()
            view_fav_ratio = avg_favs / avg_views if avg_views > 0 else 0
            print(f"    Avg views: {avg_views:.0f}")
            print(f"    Avg favorites: {avg_favs:.0f}")
            print(f"    View-to-favorite ratio: {view_fav_ratio:.3f}")

            # Sale rate
            sale_rate = (cond_data['sold'] == 'yes').sum() / len(cond_data) * 100
            print(f"    Sale success rate: {sale_rate:.1f}%")

            # Fraud hypothesis
            mint_old = cond_data[(cond_data['condition'] == 'mint') &
                                (cond_data['age_years'] > 10)]
            if condition == 'mint' and len(mint_old) > 0:
                suspicious_rate = (mint_old['suspected_counterfeit'] > 0.5).sum() / len(mint_old) * 100
                print(f"    'Mint' items >10 years old: {len(mint_old)}")
                print(f"    Suspicious (>0.5 counterfeit score): {suspicious_rate:.1f}%")

            condition_summary.append({
                'Condition': condition,
                'Avg_Price': avg_price,
                'Sale_Rate_%': sale_rate,
                'View_Fav_Ratio': view_fav_ratio
            })

        # Condition premium analysis
        print(f"\n  CONDITION PREMIUM ANALYSIS:")
        print(f"  {'─' * 70}")
        summary_df = pd.DataFrame(condition_summary)

        # Calculate multipliers relative to 'good' condition
        good_price = summary_df[summary_df['Condition'] == 'good']['Avg_Price'].values[0]
        summary_df['Price_Multiplier'] = summary_df['Avg_Price'] / good_price

        print(summary_df.to_string(index=False))

        # Category-specific condition premiums
        print(f"\n  CONDITION PREMIUM BY CATEGORY:")
        print(f"  {'─' * 70}")
        for category in ['electronics', 'clothing']:
            cat_data = self.df[self.df['category'] == category]
            print(f"\n    {category.upper()}:")
            for condition in ['mint', 'like-new', 'good', 'fair', 'poor']:
                cond_cat_data = cat_data[cat_data['condition'] == condition]
                if len(cond_cat_data) > 0:
                    avg_price = cond_cat_data['current_price'].mean()
                    print(f"      {condition:10s}: €{avg_price:.2f}")


def main():
    """Main execution function"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    analyzer = CategoricalAnalyzer(df)
    analyzer.analyze()

    print(f"\n✓ Analysis complete")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
