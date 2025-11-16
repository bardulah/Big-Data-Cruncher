"""
PHASE 1: DATA GENERATION & PROFILING
European Secondhand Marketplace Dataset Generator
Generates 10,000+ realistic marketplace listings with embedded patterns and anomalies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class MarketplaceDataGenerator:
    """Generates realistic secondhand marketplace data with correlations and anomalies"""

    def __init__(self, n_records=10000):
        self.n_records = n_records
        self.categories = ['clothing', 'electronics', 'furniture', 'books', 'collectibles']
        self.platforms = ['Vinted', 'eBay.de', 'Mercari EU', 'Facebook Marketplace', 'OLX']
        self.conditions = ['mint', 'like-new', 'good', 'fair', 'poor']

        # Brand dictionaries by category
        self.brands = {
            'clothing': ['Zara', 'H&M', 'Nike', 'Adidas', 'Gucci', 'Louis Vuitton',
                        'Prada', 'Supreme', 'Uniqlo', 'GAP', None],
            'electronics': ['Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'HP',
                           'Lenovo', 'Microsoft', 'Nintendo', 'Generic', None],
            'furniture': ['IKEA', 'Ashley', 'Wayfair', 'Herman Miller', 'West Elm',
                         'Crate&Barrel', 'Generic', None],
            'books': ['Penguin', 'HarperCollins', 'Random House', 'Vintage',
                     'Self-published', None],
            'collectibles': ['LEGO', 'Pokemon', 'Funko', 'Vintage', 'Antique',
                            'Limited Edition', None]
        }

        # Luxury brands (higher value, higher fraud risk)
        self.luxury_brands = ['Gucci', 'Louis Vuitton', 'Prada', 'Supreme',
                             'Apple', 'Herman Miller']

    def generate_base_data(self):
        """Generate base dataset with random distributions"""
        data = {
            'item_id': range(self.n_records),
            'category': np.random.choice(self.categories, self.n_records,
                                        p=[0.30, 0.25, 0.15, 0.15, 0.15]),
            'platform': np.random.choice(self.platforms, self.n_records,
                                        p=[0.25, 0.22, 0.18, 0.20, 0.15]),
        }

        df = pd.DataFrame(data)

        # Platform-specific patterns
        df['condition'] = df.apply(self._assign_condition, axis=1)
        df['brand'] = df.apply(self._assign_brand, axis=1)
        df['age_years'] = df.apply(self._assign_age, axis=1)

        # Seller characteristics (platform-dependent)
        df['seller_account_age_days'] = df.apply(self._assign_seller_age, axis=1)
        df['seller_item_count'] = df.apply(self._assign_seller_items, axis=1)
        df['seller_rating'] = df.apply(self._assign_seller_rating, axis=1)

        # Listing characteristics
        df['image_count'] = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                            self.n_records,
                                            p=[0.02, 0.05, 0.10, 0.15, 0.20,
                                               0.18, 0.12, 0.08, 0.05, 0.03, 0.02])

        df['description_length'] = np.random.gamma(2, 100, self.n_records).astype(int)
        df['description_length'] = np.clip(df['description_length'], 0, 2000)

        df['shipping_offered'] = np.random.choice([True, False], self.n_records,
                                                  p=[0.75, 0.25])

        # Pricing (category and condition dependent)
        df['original_price'] = df.apply(self._assign_original_price, axis=1)
        df['current_price'] = df.apply(self._assign_current_price, axis=1)

        # Time-based data (with seasonal patterns)
        df['month_listed'] = np.random.randint(1, 13, self.n_records)
        df['days_listed'] = df.apply(self._assign_days_listed, axis=1)

        # Engagement metrics (correlated with price, description, images)
        df['view_count'] = df.apply(self._assign_views, axis=1)
        df['favorite_count'] = df.apply(self._assign_favorites, axis=1)

        # Seller behavior
        df['price_drops'] = df.apply(self._assign_price_drops, axis=1)
        df['relistings'] = np.random.choice([0, 1, 2, 3, 4, 5], self.n_records,
                                           p=[0.60, 0.20, 0.10, 0.05, 0.03, 0.02])

        # Sale outcomes (70% unsold, 30% sold)
        df['sold'] = np.random.choice(['yes', 'no'], self.n_records, p=[0.30, 0.70])
        df = df.apply(self._adjust_sale_probability, axis=1)

        df['sale_price'] = df.apply(self._assign_sale_price, axis=1)
        df['days_to_sell'] = df.apply(self._assign_days_to_sell, axis=1)

        # Fraud indicators (correlated with multiple factors)
        df['suspected_counterfeit'] = df.apply(self._calculate_counterfeit_score, axis=1)
        df['fraud_flag'] = df['suspected_counterfeit'] > 0.75

        # Introduce missing data (~5% sparse)
        df = self._introduce_missing_data(df)

        # Add anomalies
        df = self._inject_anomalies(df)

        return df

    def _assign_condition(self, row):
        """Assign condition based on category and platform"""
        if row['platform'] == 'Vinted':  # Younger sellers, better condition
            return np.random.choice(self.conditions, p=[0.15, 0.35, 0.30, 0.15, 0.05])
        elif row['platform'] == 'eBay.de':  # More vintage items
            return np.random.choice(self.conditions, p=[0.10, 0.20, 0.35, 0.25, 0.10])
        else:
            return np.random.choice(self.conditions, p=[0.12, 0.28, 0.32, 0.20, 0.08])

    def _assign_brand(self, row):
        """Assign brand based on category"""
        category_brands = self.brands.get(row['category'], [None])
        if row['category'] == 'clothing':
            # More luxury brands in clothing
            return np.random.choice(category_brands,
                                   p=[0.15, 0.15, 0.12, 0.12, 0.03, 0.03,
                                      0.02, 0.02, 0.10, 0.10, 0.16])
        elif row['category'] == 'electronics':
            return np.random.choice(category_brands,
                                   p=[0.25, 0.20, 0.10, 0.08, 0.07, 0.06,
                                      0.06, 0.05, 0.04, 0.06, 0.03])
        else:
            return np.random.choice(category_brands)

    def _assign_age(self, row):
        """Assign item age based on category"""
        if row['category'] == 'electronics':
            # Electronics: newer items
            return int(np.clip(np.random.gamma(2, 1.5), 0, 20))
        elif row['category'] == 'collectibles':
            # Collectibles: wider age range
            return int(np.random.uniform(0, 50))
        elif row['category'] == 'furniture':
            return int(np.clip(np.random.gamma(3, 2), 0, 30))
        else:
            return int(np.clip(np.random.gamma(2.5, 2), 0, 20))

    def _assign_seller_age(self, row):
        """Assign seller account age (platform-dependent)"""
        if row['platform'] == 'Vinted':  # Younger platform, newer sellers
            return int(np.clip(np.random.gamma(2, 150), 1, 3650))
        elif row['platform'] == 'eBay.de':  # Established sellers
            return int(np.clip(np.random.gamma(4, 250), 1, 7300))
        else:
            return int(np.clip(np.random.gamma(3, 200), 1, 5475))

    def _assign_seller_items(self, row):
        """Assign seller item count (correlated with account age)"""
        base_items = row['seller_account_age_days'] / 30  # ~1 item per month
        items = int(np.clip(np.random.gamma(2, base_items/2), 1, 5000))
        return items

    def _assign_seller_rating(self, row):
        """Assign seller rating (correlated with experience)"""
        if row['seller_item_count'] > 100:  # Experienced sellers
            return np.clip(np.random.normal(4.5, 0.3), 0, 5)
        elif row['seller_item_count'] > 10:
            return np.clip(np.random.normal(4.2, 0.5), 0, 5)
        else:  # New sellers
            return np.clip(np.random.normal(3.8, 0.8), 0, 5)

    def _assign_original_price(self, row):
        """Assign original price based on category, brand, condition"""
        # Base prices by category
        base_prices = {
            'clothing': 50,
            'electronics': 300,
            'furniture': 200,
            'books': 15,
            'collectibles': 100
        }

        base = base_prices[row['category']]

        # Brand premium
        if row['brand'] in self.luxury_brands:
            base *= np.random.uniform(5, 15)  # Luxury multiplier
        elif row['brand'] is not None and row['brand'] != 'Generic':
            base *= np.random.uniform(1.5, 3)

        # Condition adjustment
        condition_mult = {
            'mint': 0.95,
            'like-new': 0.85,
            'good': 0.65,
            'fair': 0.45,
            'poor': 0.25
        }
        base *= condition_mult.get(row['condition'], 0.5)

        # Add random variation
        price = base * np.random.uniform(0.7, 1.5)
        return round(max(1, price), 2)

    def _assign_current_price(self, row):
        """Assign current price with depreciation and price drops"""
        original = row['original_price']

        # Age-based depreciation (category-specific)
        if row['category'] == 'electronics':
            # 3-5% per month depreciation
            depreciation = 1 - (row['age_years'] * 12 * np.random.uniform(0.03, 0.05))
        elif row['category'] == 'collectibles':
            # Can appreciate or depreciate
            depreciation = np.random.uniform(0.5, 2.5)
        else:
            depreciation = 1 - (row['age_years'] * 0.05)  # 5% per year

        current = original * max(0.1, depreciation)

        # Random price adjustment
        current *= np.random.uniform(0.8, 1.1)

        return round(max(1, current), 2)

    def _assign_days_listed(self, row):
        """Assign days listed (influenced by multiple factors)"""
        base_days = np.random.gamma(3, 15)

        # Longer for overpriced items (we'll adjust later)
        # Shorter for well-priced items
        days = int(np.clip(base_days, 1, 365))
        return days

    def _assign_views(self, row):
        """Assign view count (correlated with price, images, description)"""
        base_views = 50

        # Image count effect
        base_views *= (1 + row['image_count'] * 0.15)

        # Description length effect (diminishing returns)
        desc_boost = min(1.5, 1 + row['description_length'] / 1000)
        base_views *= desc_boost

        # Luxury items get more views
        if row['brand'] in self.luxury_brands:
            base_views *= np.random.uniform(3, 7)

        # Category effects
        if row['category'] == 'electronics':
            base_views *= 1.5
        elif row['category'] == 'books':
            base_views *= 0.6

        # Days listed effect
        base_views *= (1 + row['days_listed'] / 100)

        views = int(np.clip(np.random.poisson(base_views), 0, 10000))
        return views

    def _assign_favorites(self, row):
        """Assign favorite count (fraction of views, price-dependent)"""
        # Favorite rate: typically 5-20% of views
        fav_rate = np.random.uniform(0.05, 0.20)

        # Lower favorite rate for overpriced items
        # Higher for well-priced items

        favorites = int(row['view_count'] * fav_rate)
        return favorites

    def _assign_price_drops(self, row):
        """Assign number of price drops"""
        # More price drops for items listed longer
        prob_drop = min(0.8, row['days_listed'] / 100)

        if np.random.random() < prob_drop:
            return np.random.choice([1, 2, 3, 4, 5], p=[0.40, 0.30, 0.15, 0.10, 0.05])
        return 0

    def _adjust_sale_probability(self, row):
        """Adjust sale probability based on multiple factors"""
        # Base probability: 30% sold
        sell_prob = 0.30

        # Good pricing increases probability
        if row['current_price'] < row['original_price'] * 0.7:
            sell_prob += 0.25

        # Good seller rating helps
        if row['seller_rating'] > 4.0:
            sell_prob += 0.15

        # Images help
        if row['image_count'] >= 3:
            sell_prob += 0.10

        # Good description helps
        if row['description_length'] > 200:
            sell_prob += 0.10

        # Platform effects
        if row['platform'] == 'eBay.de':
            sell_prob += 0.05

        # Price drops indicate desperation but also price correction
        if row['price_drops'] > 2:
            sell_prob += 0.15

        sell_prob = min(0.95, sell_prob)

        # Update sold status
        row['sold'] = 'yes' if np.random.random() < sell_prob else 'no'
        return row

    def _assign_sale_price(self, row):
        """Assign final sale price if sold"""
        if row['sold'] == 'yes':
            # Usually sell at current price or slightly lower
            sale = row['current_price'] * np.random.uniform(0.85, 1.0)
            return round(max(1, sale), 2)
        return np.nan

    def _assign_days_to_sell(self, row):
        """Assign days to sell if sold"""
        if row['sold'] == 'yes':
            # Sold items typically sell faster than average listing duration
            days = int(row['days_listed'] * np.random.uniform(0.3, 0.8))
            return max(1, days)
        return np.nan

    def _calculate_counterfeit_score(self, row):
        """Calculate counterfeit probability score (0-1)"""
        score = 0.0

        # Luxury brand at low price
        if row['brand'] in self.luxury_brands:
            if row['current_price'] < row['original_price'] * 0.3:
                score += 0.4
            else:
                score += 0.05

        # New seller with luxury items
        if row['seller_account_age_days'] < 30 and row['brand'] in self.luxury_brands:
            score += 0.3

        # Mint condition on old item (suspicious)
        if row['condition'] == 'mint' and row['age_years'] > 10:
            score += 0.2

        # No images but luxury
        if row['image_count'] == 0 and row['brand'] in self.luxury_brands:
            score += 0.15

        # Very short description
        if row['description_length'] < 20:
            score += 0.1

        # Multiple items from new seller
        if row['seller_account_age_days'] < 10 and row['seller_item_count'] > 50:
            score += 0.25

        # Random noise
        score += np.random.uniform(-0.1, 0.1)

        return max(0, min(1, score))

    def _introduce_missing_data(self, df):
        """Introduce ~5% missing data in realistic patterns"""
        # Missing seller ratings for new sellers
        new_seller_mask = df['seller_account_age_days'] < 30
        df.loc[new_seller_mask & (np.random.random(len(df)) < 0.3), 'seller_rating'] = np.nan

        # Missing brand for generic items
        df.loc[np.random.random(len(df)) < 0.05, 'brand'] = np.nan

        # Missing original price sometimes
        df.loc[np.random.random(len(df)) < 0.02, 'original_price'] = np.nan

        return df

    def _inject_anomalies(self, df):
        """Inject specific anomalies for testing detection"""
        n_anomalies = 100
        anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

        for idx in anomaly_indices[:20]:
            # Super low price luxury items (counterfeits)
            df.loc[idx, 'brand'] = np.random.choice(self.luxury_brands)
            df.loc[idx, 'current_price'] = df.loc[idx, 'original_price'] * 0.15
            df.loc[idx, 'suspected_counterfeit'] = 0.95

        for idx in anomaly_indices[20:40]:
            # Extremely high prices (overpriced or rare)
            df.loc[idx, 'current_price'] = df.loc[idx, 'original_price'] * 5.0
            df.loc[idx, 'view_count'] = int(df.loc[idx, 'view_count'] * 3)
            df.loc[idx, 'favorite_count'] = 0  # High views but no favorites

        for idx in anomaly_indices[40:60]:
            # New sellers with bulk luxury listings
            df.loc[idx, 'seller_account_age_days'] = np.random.randint(1, 10)
            df.loc[idx, 'seller_item_count'] = np.random.randint(100, 500)
            df.loc[idx, 'brand'] = np.random.choice(self.luxury_brands)
            df.loc[idx, 'suspected_counterfeit'] = 0.88

        for idx in anomaly_indices[60:80]:
            # Suspicious behavioral patterns
            df.loc[idx, 'view_count'] = np.random.randint(1000, 5000)
            df.loc[idx, 'favorite_count'] = 0
            df.loc[idx, 'days_listed'] = np.random.randint(150, 300)
            df.loc[idx, 'price_drops'] = 5

        for idx in anomaly_indices[80:]:
            # Data quality issues
            df.loc[idx, 'image_count'] = 0
            df.loc[idx, 'description_length'] = np.random.randint(1, 15)
            df.loc[idx, 'days_listed'] = np.random.randint(200, 365)

        return df

    def calculate_quality_metrics(self, df):
        """Calculate data quality metrics"""
        print("\n" + "="*80)
        print("DATA QUALITY METRICS")
        print("="*80)

        # Basic stats
        print(f"\nDataset size: {len(df):,} records")
        print(f"Number of features: {len(df.columns)}")

        # Missing data analysis
        print("\nMissing Data Analysis:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing_Count': missing[missing > 0],
            'Missing_Percent': missing_pct[missing > 0]
        }).sort_values('Missing_Percent', ascending=False)
        print(missing_df)

        # Distribution analysis
        print("\nSale Distribution:")
        print(df['sold'].value_counts(normalize=True).round(3))

        print("\nCategory Distribution:")
        print(df['category'].value_counts(normalize=True).round(3))

        print("\nPlatform Distribution:")
        print(df['platform'].value_counts(normalize=True).round(3))

        print("\nCondition Distribution:")
        print(df['condition'].value_counts(normalize=True).round(3))

        # Correlation matrix for numeric variables
        print("\nKey Pearson Correlations:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    strong_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })

        strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation',
                                                               ascending=False,
                                                               key=abs)
        print(strong_corr_df.head(15).to_string(index=False))

        # Outlier detection preview
        print("\nOutlier Detection Preview (IQR Method):")
        for col in ['current_price', 'view_count', 'days_listed']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

        # Fraud indicators
        print(f"\nFraud Indicators:")
        print(f"  High counterfeit risk (>0.75): {(df['suspected_counterfeit'] > 0.75).sum()}")
        print(f"  Fraud flags: {df['fraud_flag'].sum()}")
        print(f"  Luxury items: {df['brand'].isin(self.luxury_brands).sum()}")

        return corr_matrix


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("PHASE 1: MARKETPLACE DATA GENERATION ENGINE")
    print("European Secondhand Marketplace - 10,000 Records")
    print("="*80)

    # Generate data
    print("\nGenerating synthetic dataset...")
    generator = MarketplaceDataGenerator(n_records=10000)
    df = generator.generate_base_data()

    print(f"✓ Generated {len(df):,} records with {len(df.columns)} features")

    # Display sample
    print("\n" + "-"*80)
    print("SAMPLE DATA (First 10 Records):")
    print("-"*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(10))

    # Calculate quality metrics
    corr_matrix = generator.calculate_quality_metrics(df)

    # Save dataset
    output_file = 'marketplace_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to: {output_file}")

    # Save correlation matrix
    corr_matrix.to_csv('correlation_matrix.csv')
    print(f"✓ Correlation matrix saved to: correlation_matrix.csv")

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)

    return df, corr_matrix


if __name__ == "__main__":
    df, corr = main()
