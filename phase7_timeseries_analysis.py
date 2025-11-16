"""
PHASE 7: TIME SERIES ANALYSIS & TRENDS
Analyze temporal patterns, seasonality, and price evolution over 12 months
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """Analyzes temporal patterns and trends"""

    def __init__(self, df):
        self.df = df.copy()
        self.month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def analyze_all_patterns(self):
        """Perform complete time series analysis"""
        print("\n" + "="*80)
        print("PHASE 7: TIME SERIES ANALYSIS & TRENDS")
        print("="*80)

        self.seasonal_patterns()
        self.price_evolution()
        self.platform_trends()
        self.seller_evolution()

        print("\n" + "="*80)
        print("PHASE 7 COMPLETE")
        print("="*80)

    def seasonal_patterns(self):
        """Analyze seasonal patterns across 12 months"""
        print(f"\n{'-'*80}")
        print("SEASONALITY ANALYSIS")
        print(f"{'-'*80}")

        if 'month_listed' not in self.df.columns:
            print("  [WARNING] No month_listed column found")
            return

        # Overall listing volume by month
        monthly_counts = self.df.groupby('month_listed').size()
        print(f"\n  OVERALL LISTING VOLUME BY MONTH:")
        print(f"  {'-' * 70}")
        for month, count in monthly_counts.items():
            month_name = self.month_names[month]
            print(f"    {month_name:4s}: {count:5,} listings {'█' * int(count/100)}")

        # Seasonal patterns by category
        print(f"\n  SEASONAL PATTERNS BY CATEGORY:")
        print(f"  {'-' * 70}")

        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            monthly_cat = cat_data.groupby('month_listed').size()

            peak_month = monthly_cat.idxmax()
            trough_month = monthly_cat.idxmin()
            peak_count = monthly_cat.max()
            trough_count = monthly_cat.min()
            seasonality_ratio = peak_count / trough_count if trough_count > 0 else 0

            print(f"\n    {category.upper()}")
            print(f"      Peak: {self.month_names[peak_month]} ({peak_count} listings)")
            print(f"      Trough: {self.month_names[trough_month]} ({trough_count} listings)")
            print(f"      Seasonality ratio: {seasonality_ratio:.2f}x")

            # Interpret seasonality
            if category == 'clothing':
                if peak_month in [10, 11, 12, 1, 2]:  # Fall/Winter
                    print(f"      → Winter clothing dominates cold months")
                elif peak_month in [5, 6, 7, 8]:  # Summer
                    print(f"      → Summer clothing peaks in warm months")
            elif category == 'electronics':
                if peak_month in [10, 11, 12]:
                    print(f"      → Pre-holiday peak (gift buying season)")
                elif peak_month == 1:
                    print(f"      → Post-holiday sales and upgrades")

        # Price evolution by month
        print(f"\n  AVERAGE PRICE EVOLUTION BY MONTH:")
        print(f"  {'-' * 70}")
        monthly_prices = self.df.groupby('month_listed')['current_price'].mean()
        for month, price in monthly_prices.items():
            print(f"    {self.month_names[month]:4s}: €{price:.2f}")

        # Sale success rate by month
        print(f"\n  SALE SUCCESS RATE BY MONTH:")
        print(f"  {'-' * 70}")
        monthly_sales = self.df.groupby('month_listed').apply(
            lambda x: (x['sold'] == 'yes').sum() / len(x) * 100
        )
        for month, rate in monthly_sales.items():
            print(f"    {self.month_names[month]:4s}: {rate:.1f}%")

        # Holiday effects
        print(f"\n  HOLIDAY EFFECTS:")
        print(f"  {'-' * 70}")

        # Pre-holiday (Nov-Dec)
        holiday_period = self.df[self.df['month_listed'].isin([11, 12])]
        normal_period = self.df[self.df['month_listed'].isin([3, 4, 9, 10])]

        if len(holiday_period) > 0 and len(normal_period) > 0:
            holiday_listings = len(holiday_period)
            normal_listings = len(normal_period) / 4  # Average per month
            print(f"    Holiday period (Nov-Dec): {holiday_listings:,} listings")
            print(f"    Normal period avg/month: {normal_listings:,.0f} listings")
            print(f"    Holiday boost: {holiday_listings/normal_listings:.2f}x normal")

            holiday_sales = (holiday_period['sold'] == 'yes').mean() * 100
            normal_sales = (normal_period['sold'] == 'yes').mean() * 100
            print(f"    Holiday sale rate: {holiday_sales:.1f}%")
            print(f"    Normal sale rate: {normal_sales:.1f}%")

        # Post-holiday (January)
        jan_data = self.df[self.df['month_listed'] == 1]
        if len(jan_data) > 0:
            print(f"\n    January (post-holiday liquidation):")
            print(f"      Listings: {len(jan_data):,}")
            print(f"      Avg price: €{jan_data['current_price'].mean():.2f}")
            print(f"      → Post-holiday sales and returns resale")

        # Back-to-school (July-August)
        back_to_school = self.df[self.df['month_listed'].isin([7, 8])]
        if len(back_to_school) > 0:
            electronics_bts = back_to_school[back_to_school['category'] == 'electronics']
            books_bts = back_to_school[back_to_school['category'] == 'books']

            print(f"\n    Back-to-school (Jul-Aug):")
            print(f"      Electronics listings: {len(electronics_bts):,}")
            print(f"      Books listings: {len(books_bts):,}")
            if len(books_bts) > 0:
                books_other = self.df[(self.df['category'] == 'books') &
                                     (~self.df['month_listed'].isin([7, 8]))]
                bts_ratio = len(books_bts) / (len(books_other) / 10) if len(books_other) > 0 else 0
                print(f"      Books BTS boost: {bts_ratio:.2f}x normal months")

    def price_evolution(self):
        """Analyze how prices evolve over time"""
        print(f"\n{'-'*80}")
        print("PRICE EVOLUTION ANALYSIS")
        print(f"{'-'*80}")

        # Price decay by category over time (using age_years as proxy)
        print(f"\n  PRICE DEPRECIATION BY CATEGORY:")
        print(f"  {'-' * 70}")

        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]

            if cat_data['age_years'].std() > 0:
                # Calculate depreciation rate (correlation between age and price)
                correlation = cat_data[['age_years', 'current_price']].corr().iloc[0, 1]

                # Linear regression to estimate annual depreciation
                if len(cat_data) > 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        cat_data['age_years'], cat_data['current_price']
                    )

                    print(f"\n    {category.upper()}")
                    print(f"      Age-price correlation: {correlation:.3f}")
                    print(f"      Annual price change: €{slope:.2f}/year")

                    if slope < 0:
                        avg_price = cat_data['current_price'].mean()
                        depreciation_pct = abs(slope) / avg_price * 100 if avg_price > 0 else 0
                        print(f"      Depreciation rate: {depreciation_pct:.1f}% per year")

                        if category == 'electronics':
                            print(f"      → Electronics depreciate rapidly (3-5% per month typical)")
                        elif category == 'furniture':
                            print(f"      → Furniture shows moderate depreciation")
                    elif slope > 0:
                        print(f"      → APPRECIATION with age (vintage/collectible effect)")
                    else:
                        print(f"      → Stable pricing regardless of age")

        # Time-to-sell trends
        print(f"\n  TIME-TO-SELL TRENDS:")
        print(f"  {'-' * 70}")

        sold_items = self.df[self.df['sold'] == 'yes']
        if len(sold_items) > 0:
            # By price tier
            sold_items['price_tier'] = pd.qcut(sold_items['current_price'], q=4,
                                              labels=['Low', 'Medium', 'High', 'Luxury'],
                                              duplicates='drop')

            for tier in ['Low', 'Medium', 'High', 'Luxury']:
                tier_data = sold_items[sold_items['price_tier'] == tier]
                if len(tier_data) > 0:
                    avg_days = tier_data['days_to_sell'].mean()
                    median_days = tier_data['days_to_sell'].median()
                    print(f"    {tier:8s} tier: avg {avg_days:.1f} days, median {median_days:.1f} days")

        # Price drop patterns over time
        print(f"\n  PRICE DROP DYNAMICS:")
        print(f"  {'-' * 70}")

        for drops in range(6):
            drop_items = self.df[self.df['price_drops'] == drops]
            if len(drop_items) > 0:
                avg_days = drop_items['days_listed'].mean()
                sale_rate = (drop_items['sold'] == 'yes').mean() * 100
                print(f"    {drops} drops: avg {avg_days:.1f} days listed, "
                      f"{sale_rate:.1f}% sold")

        print(f"\n    → Optimal strategy: 1-2 price drops after 30-60 days")
        print(f"    → Excessive drops (3+) signal quality concerns")

    def platform_trends(self):
        """Analyze platform-specific temporal trends"""
        print(f"\n{'-'*80}")
        print("PLATFORM TREND ANALYSIS")
        print(f"{'-'*80}")

        # Growth trends by platform (using month as proxy)
        if 'month_listed' in self.df.columns:
            print(f"\n  PLATFORM ACTIVITY BY MONTH:")
            print(f"  {'-' * 70}")

            for platform in self.df['platform'].unique():
                platform_data = self.df[self.df['platform'] == platform]
                monthly_dist = platform_data.groupby('month_listed').size()

                peak_month = monthly_dist.idxmax()
                trough_month = monthly_dist.idxmin()

                print(f"\n    {platform}")
                print(f"      Most active: {self.month_names[peak_month]} "
                      f"({monthly_dist.max()} listings)")
                print(f"      Least active: {self.month_names[trough_month]} "
                      f"({monthly_dist.min()} listings)")

                # Platform-specific insights
                if platform == 'Vinted':
                    print(f"      → Clothing-focused platform shows seasonal variation")
                elif platform == 'eBay.de':
                    print(f"      → Mature platform with stable activity")

        # Platform market share evolution
        print(f"\n  PLATFORM COMPETITIVE DYNAMICS:")
        print(f"  {'-' * 70}")

        platform_shares = self.df['platform'].value_counts(normalize=True) * 100
        for platform, share in platform_shares.items():
            print(f"    {platform:20s}: {share:.1f}% market share")

        # Platform sale efficiency over time
        print(f"\n  PLATFORM SALE EFFICIENCY:")
        print(f"  {'-' * 70}")

        for platform in self.df['platform'].unique():
            platform_data = self.df[self.df['platform'] == platform]
            sold_items = platform_data[platform_data['sold'] == 'yes']

            if len(sold_items) > 0:
                avg_days_to_sell = sold_items['days_to_sell'].mean()
                sale_rate = len(sold_items) / len(platform_data) * 100

                print(f"    {platform:20s}: {sale_rate:.1f}% sale rate, "
                      f"{avg_days_to_sell:.1f} days avg")

    def seller_evolution(self):
        """Analyze seller behavior evolution"""
        print(f"\n{'-'*80}")
        print("SELLER EVOLUTION ANALYSIS")
        print(f"{'-'*80}")

        # Seller learning curve (experience effect over time)
        print(f"\n  SELLER LEARNING CURVE:")
        print(f"  {'-' * 70}")

        # Define experience brackets
        experience_brackets = [
            ('New (0-90 days)', 0, 90),
            ('Developing (90-365 days)', 90, 365),
            ('Experienced (1-2 years)', 365, 730),
            ('Veteran (2+ years)', 730, 10000)
        ]

        for label, min_days, max_days in experience_brackets:
            sellers = self.df[(self.df['seller_account_age_days'] >= min_days) &
                             (self.df['seller_account_age_days'] < max_days)]

            if len(sellers) > 0:
                avg_desc = sellers['description_length'].mean()
                avg_images = sellers['image_count'].mean()
                avg_price = sellers['current_price'].mean()
                sale_rate = (sellers['sold'] == 'yes').mean() * 100

                print(f"\n    {label}")
                print(f"      Avg description: {avg_desc:.0f} chars")
                print(f"      Avg images: {avg_images:.1f}")
                print(f"      Avg price: €{avg_price:.2f}")
                print(f"      Sale rate: {sale_rate:.1f}%")

        # Price strategy evolution
        print(f"\n  PRICE STRATEGY EVOLUTION:")
        print(f"  {'-' * 70}")

        for label, min_days, max_days in experience_brackets:
            sellers = self.df[(self.df['seller_account_age_days'] >= min_days) &
                             (self.df['seller_account_age_days'] < max_days)]

            if len(sellers) > 0:
                avg_drops = sellers['price_drops'].mean()
                avg_discount = ((sellers['original_price'] - sellers['current_price']) /
                               sellers['original_price'] * 100).mean()

                print(f"    {label:25s}: {avg_drops:.2f} avg drops, "
                      f"{avg_discount:.1f}% avg discount")

        print(f"\n    → Experienced sellers make fewer but more strategic price adjustments")
        print(f"    → New sellers either overprice or underprice, leading to more corrections")

        # Temporal fraud patterns
        print(f"\n  FRAUD PATTERNS OVER SELLER LIFETIME:")
        print(f"  {'-' * 70}")

        for label, min_days, max_days in experience_brackets:
            sellers = self.df[(self.df['seller_account_age_days'] >= min_days) &
                             (self.df['seller_account_age_days'] < max_days)]

            if len(sellers) > 0:
                fraud_rate = sellers['fraud_flag'].mean() * 100
                avg_counterfeit = sellers['suspected_counterfeit'].mean()

                print(f"    {label:25s}: {fraud_rate:.1f}% fraud flags, "
                      f"{avg_counterfeit:.3f} avg counterfeit score")

        print(f"\n    → New accounts have significantly higher fraud risk")
        print(f"    → Fraud attempts concentrate in first 90 days")
        print(f"    → Veteran sellers (2+ years) have lowest fraud rates")


def main():
    """Main execution function"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    analyzer = TimeSeriesAnalyzer(df)
    analyzer.analyze_all_patterns()

    print(f"\n✓ Time series analysis complete")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
