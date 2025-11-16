"""
PHASE 8: NETWORK ANALYSIS & FRAUD RING DETECTION
Analyzes seller relationships and detects coordinated fraud networks
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class NetworkAnalyzer:
    """Detects seller networks and fraud rings"""

    def __init__(self, df):
        self.df = df.copy()

    def analyze_all_networks(self):
        """Perform complete network analysis"""
        print("\n" + "="*80)
        print("PHASE 8: NETWORK ANALYSIS & FRAUD RING DETECTION")
        print("="*80)

        # Seller similarity networks
        self.detect_seller_clusters()

        # Fraud ring detection
        self.detect_fraud_rings()

        # Platform cross-listing patterns
        self.analyze_cross_platform_patterns()

        # Pricing collusion detection
        self.detect_price_collusion()

        print("\n" + "="*80)
        print("PHASE 8 COMPLETE")
        print("="*80)

    def detect_seller_clusters(self):
        """Find sellers with similar behavioral patterns"""
        print(f"\n{'-'*80}")
        print("SELLER CLUSTERING & SIMILARITY NETWORKS")
        print(f"{'-'*80}")

        # Group sellers by account age bucket
        self.df['seller_bucket'] = pd.cut(
            self.df['seller_account_age_days'],
            bins=[0, 7, 30, 90, 365, 10000],
            labels=['<7d', '7-30d', '30-90d', '90-365d', '365d+']
        )

        # Find suspicious seller groups
        seller_groups = self.df.groupby('seller_account_age_days').agg({
            'item_id': 'count',
            'suspected_counterfeit': 'mean',
            'current_price': 'mean',
            'fraud_flag': 'sum'
        }).reset_index()

        seller_groups.columns = ['seller_age', 'item_count', 'avg_counterfeit',
                                 'avg_price', 'fraud_count']

        # Suspicious groups: many items, high fraud
        suspicious = seller_groups[
            (seller_groups['item_count'] > 50) &
            (seller_groups['avg_counterfeit'] > 0.3)
        ]

        print(f"\n  Suspicious Seller Groups Identified: {len(suspicious)}")

        if len(suspicious) > 0:
            print(f"\n  TOP 10 SUSPICIOUS SELLER GROUPS:")
            print(f"  {'-' * 75}")

            for idx, row in suspicious.nlargest(10, 'avg_counterfeit').iterrows():
                print(f"    Seller age {row['seller_age']}d: "
                      f"{row['item_count']} items, "
                      f"{row['avg_counterfeit']:.3f} avg counterfeit score")

        # Network metrics
        print(f"\n  NETWORK STATISTICS:")
        print(f"    Total unique seller ages: {self.df['seller_account_age_days'].nunique()}")
        print(f"    Sellers with 100+ items: {(self.df.groupby('seller_account_age_days')['item_id'].count() > 100).sum()}")
        print(f"    Potential fraud networks: {len(suspicious)}")

    def detect_fraud_rings(self):
        """Detect coordinated fraud operations"""
        print(f"\n{'-'*80}")
        print("FRAUD RING DETECTION")
        print(f"{'-'*80}")

        # Pattern 1: Multiple new sellers with identical pricing patterns
        new_sellers = self.df[self.df['seller_account_age_days'] < 30]
        luxury_new = new_sellers[new_sellers['brand'].isin(['Gucci', 'Louis Vuitton', 'Prada', 'Supreme'])]

        if len(luxury_new) > 0:
            # Group by price (rounded to nearest 10)
            luxury_new['price_rounded'] = (luxury_new['current_price'] / 10).round() * 10

            price_groups = luxury_new.groupby('price_rounded').agg({
                'item_id': 'count',
                'seller_account_age_days': lambda x: list(x.unique())
            })

            suspicious_groups = price_groups[price_groups['item_id'] > 5]

            print(f"\n  SUSPECTED FRAUD RINGS:")
            print(f"    Coordinated pricing groups: {len(suspicious_groups)}")

            if len(suspicious_groups) > 0:
                print(f"\n  Details:")
                for price, row in suspicious_groups.head(5).iterrows():
                    print(f"    €{price:.0f} price point: {row['item_id']} sellers")

        # Pattern 2: Same-day account creation with similar items
        same_day_sellers = self.df.groupby('seller_account_age_days').agg({
            'item_id': 'count',
            'category': lambda x: x.mode()[0] if len(x) > 0 else None,
            'suspected_counterfeit': 'mean'
        })

        coordinated = same_day_sellers[
            (same_day_sellers['item_id'] > 20) &
            (same_day_sellers['suspected_counterfeit'] > 0.5)
        ]

        print(f"\n  COORDINATED ACCOUNT CREATION:")
        print(f"    Suspicious same-day seller groups: {len(coordinated)}")

        if len(coordinated) > 0:
            for seller_age, row in coordinated.head(5).iterrows():
                print(f"    Day {seller_age}: {row['item_id']} accounts, "
                      f"category: {row['category']}, "
                      f"fraud score: {row['suspected_counterfeit']:.3f}")

    def analyze_cross_platform_patterns(self):
        """Detect sellers active across multiple platforms"""
        print(f"\n{'-'*80}")
        print("CROSS-PLATFORM PATTERN ANALYSIS")
        print(f"{'-'*80}")

        # Analyze by seller characteristics (using account age as proxy for same seller)
        # In real system, would use seller_id

        platform_diversity = self.df.groupby('seller_account_age_days')['platform'].nunique()
        multi_platform = platform_diversity[platform_diversity > 1]

        print(f"\n  Multi-platform sellers: {len(multi_platform)}")

        # Price arbitrage detection (same item, different platforms, different prices)
        # Group similar items
        self.df['item_signature'] = (
            self.df['category'] + '_' +
            self.df['brand'].fillna('none') + '_' +
            self.df['condition'] + '_' +
            (self.df['age_years'] // 2).astype(str)  # Bucket age
        )

        arbitrage_candidates = []

        for signature in self.df['item_signature'].unique():
            items = self.df[self.df['item_signature'] == signature]

            if len(items) > 1:
                platforms = items.groupby('platform')['current_price'].mean()

                if len(platforms) > 1:
                    price_range = platforms.max() - platforms.min()
                    if price_range > platforms.mean() * 0.5:  # 50%+ price difference
                        arbitrage_candidates.append({
                            'signature': signature,
                            'platforms': len(platforms),
                            'price_range': price_range,
                            'min_price': platforms.min(),
                            'max_price': platforms.max()
                        })

        print(f"\n  PRICE ARBITRAGE OPPORTUNITIES:")
        print(f"    Items with >50% price difference across platforms: {len(arbitrage_candidates)}")

        if len(arbitrage_candidates) > 0:
            print(f"\n  Top 5 arbitrage opportunities:")
            arb_df = pd.DataFrame(arbitrage_candidates).nlargest(5, 'price_range')

            for idx, row in arb_df.iterrows():
                print(f"    {row['signature']}: €{row['min_price']:.2f} - €{row['max_price']:.2f} "
                      f"(€{row['price_range']:.2f} difference)")

    def detect_price_collusion(self):
        """Detect potential price fixing among sellers"""
        print(f"\n{'-'*80}")
        print("PRICE COLLUSION DETECTION")
        print(f"{'-'*80}")

        # Look for unusually consistent pricing within categories
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]

            # Calculate price variance
            price_cv = cat_data['current_price'].std() / cat_data['current_price'].mean()

            # Low variance might indicate price fixing
            if price_cv < 0.3:  # Less than 30% coefficient of variation
                print(f"\n  Potential collusion in {category}:")
                print(f"    Price consistency: CV = {price_cv:.3f} (unusually low variance)")
                print(f"    Mean price: €{cat_data['current_price'].mean():.2f}")

        # Check for synchronized price changes
        # Items with same price_drops on same days_listed
        drop_sync = self.df.groupby(['days_listed', 'price_drops']).size()
        suspicious_sync = drop_sync[drop_sync > 50]

        if len(suspicious_sync) > 0:
            print(f"\n  SYNCHRONIZED PRICING BEHAVIOR:")
            print(f"    Suspiciously coordinated price changes: {len(suspicious_sync)} patterns")


def main():
    """Main execution"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    analyzer = NetworkAnalyzer(df)
    analyzer.analyze_all_networks()

    print(f"\n✓ Network analysis complete")


if __name__ == "__main__":
    main()
