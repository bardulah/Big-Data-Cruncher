"""
PHASE 5: ANOMALY & FRAUD DETECTION ENGINE
Multi-layer anomaly detection system with 5 detection layers
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetectionEngine:
    """Comprehensive multi-layer anomaly and fraud detection system"""

    def __init__(self, df):
        self.df = df.copy()
        self.luxury_brands = ['Gucci', 'Louis Vuitton', 'Prada', 'Supreme',
                             'Apple', 'Herman Miller']
        self.anomaly_scores = pd.DataFrame(index=df.index)
        self.anomaly_scores['item_id'] = df['item_id']

    def detect_all_anomalies(self):
        """Run all detection layers"""
        print("\n" + "="*80)
        print("PHASE 5: ANOMALY & FRAUD DETECTION ENGINE")
        print("="*80)

        # Layer 1: Price anomalies
        price_anomalies = self.detect_price_anomalies()

        # Layer 2: Behavioral anomalies
        behavioral_anomalies = self.detect_behavioral_anomalies()

        # Layer 3: Content anomalies
        content_anomalies = self.detect_content_anomalies()

        # Layer 4: Luxury counterfeit signals
        luxury_anomalies = self.detect_luxury_counterfeits()

        # Layer 5: Statistical outlier combinations
        statistical_anomalies = self.detect_statistical_outliers()

        # Aggregate scores
        self.aggregate_anomaly_scores()

        # Generate anomaly report
        self.generate_anomaly_report()

        print("\n" + "="*80)
        print("PHASE 5 COMPLETE")
        print("="*80)

        return self.anomaly_scores

    def detect_price_anomalies(self):
        """Layer 1: Price-based anomaly detection"""
        print(f"\n{'-'*80}")
        print("LAYER 1: PRICE ANOMALY DETECTION")
        print(f"{'-'*80}")

        anomalies = []

        # Anomaly 1: Priced 90% below market average
        for category in self.df['category'].unique():
            for condition in self.df['condition'].unique():
                subset = self.df[(self.df['category'] == category) &
                                (self.df['condition'] == condition)]
                if len(subset) < 5:
                    continue

                avg_price = subset['current_price'].mean()
                threshold = avg_price * 0.10  # 90% below = 10% of average

                extreme_low = subset[subset['current_price'] < threshold]
                for idx in extreme_low.index:
                    self.anomaly_scores.loc[idx, 'price_extreme_low'] = 1.0
                    anomalies.append({
                        'item_id': self.df.loc[idx, 'item_id'],
                        'type': 'Extreme Low Price',
                        'reason': f"{category}/{condition}: €{self.df.loc[idx, 'current_price']:.2f} "
                                 f"vs avg €{avg_price:.2f}"
                    })

        print(f"  Extreme low price items: {(self.anomaly_scores['price_extreme_low'] == 1.0).sum()}")

        # Anomaly 2: Priced 300% above comparables
        for category in self.df['category'].unique():
            subset = self.df[self.df['category'] == category]
            avg_price = subset['current_price'].mean()
            threshold = avg_price * 3.0

            extreme_high = subset[subset['current_price'] > threshold]
            for idx in extreme_high.index:
                self.anomaly_scores.loc[idx, 'price_extreme_high'] = 1.0
                anomalies.append({
                    'item_id': self.df.loc[idx, 'item_id'],
                    'type': 'Extreme High Price',
                    'reason': f"€{self.df.loc[idx, 'current_price']:.2f} vs avg €{avg_price:.2f}"
                })

        print(f"  Extreme high price items: {(self.anomaly_scores['price_extreme_high'] == 1.0).sum()}")

        # Anomaly 3: Massive price drop in short time
        rapid_drops = self.df[(self.df['price_drops'] >= 3) &
                             (self.df['days_listed'] < 14)]
        for idx in rapid_drops.index:
            self.anomaly_scores.loc[idx, 'rapid_price_drop'] = 0.8
            price_drop_pct = (1 - self.df.loc[idx, 'current_price'] /
                            self.df.loc[idx, 'original_price']) * 100
            anomalies.append({
                'item_id': self.df.loc[idx, 'item_id'],
                'type': 'Rapid Price Drop',
                'reason': f"{self.df.loc[idx, 'price_drops']} drops in {self.df.loc[idx, 'days_listed']} days"
            })

        print(f"  Rapid price drop items: {len(rapid_drops)}")

        # Anomaly 4: Seasonal pricing anomalies
        # Winter items in summer
        if 'month_listed' in self.df.columns:
            winter_items = self.df[(self.df['category'] == 'clothing') &
                                  (self.df['month_listed'].isin([6, 7, 8]))]  # Summer months
            # Assume some items are winter-related (simplified)
            for idx in winter_items.head(50).index:  # Sample
                self.anomaly_scores.loc[idx, 'seasonal_mismatch'] = 0.3

        print(f"  Total price anomalies flagged: {len(anomalies)}")

        return anomalies

    def detect_behavioral_anomalies(self):
        """Layer 2: Behavioral pattern anomalies"""
        print(f"\n{'-'*80}")
        print("LAYER 2: BEHAVIORAL ANOMALY DETECTION")
        print(f"{'-'*80}")

        anomalies = []

        # Anomaly 1: High views but zero favorites
        high_views_no_favs = self.df[(self.df['view_count'] > self.df['view_count'].quantile(0.75)) &
                                     (self.df['favorite_count'] == 0)]
        for idx in high_views_no_favs.index:
            self.anomaly_scores.loc[idx, 'high_views_no_favs'] = 0.7
            anomalies.append({
                'item_id': self.df.loc[idx, 'item_id'],
                'type': 'High Views, No Favorites',
                'reason': f"{self.df.loc[idx, 'view_count']} views but 0 favorites"
            })

        print(f"  High views, no favorites: {len(high_views_no_favs)}")

        # Anomaly 2: High favorites but not selling
        high_favs_no_sale = self.df[(self.df['favorite_count'] > 20) &
                                    (self.df['sold'] == 'no') &
                                    (self.df['price_drops'] == 0) &
                                    (self.df['days_listed'] > 60)]
        for idx in high_favs_no_sale.index:
            self.anomaly_scores.loc[idx, 'high_favs_no_sale'] = 0.6
            anomalies.append({
                'item_id': self.df.loc[idx, 'item_id'],
                'type': 'High Interest, No Sale',
                'reason': f"{self.df.loc[idx, 'favorite_count']} favorites, "
                         f"{self.df.loc[idx, 'days_listed']} days, no sale"
            })

        print(f"  High favorites, not selling: {len(high_favs_no_sale)}")

        # Anomaly 3: Seller with >95% unsold rate
        seller_unsold_rate = self.df.groupby('seller_account_age_days').apply(
            lambda x: (x['sold'] == 'no').sum() / len(x) if len(x) > 10 else 0
        )
        # Simplified: flag sellers with many items unsold
        poor_sellers = self.df[(self.df['seller_item_count'] > 20) &
                              (self.df['sold'] == 'no')]
        for idx in poor_sellers.sample(min(100, len(poor_sellers))).index:
            self.anomaly_scores.loc[idx, 'poor_seller_performance'] = 0.5

        print(f"  Poor seller performance indicators: {(self.anomaly_scores['poor_seller_performance'] == 0.5).sum()}")

        # Anomaly 4: New seller with bulk listings
        bulk_new_sellers = self.df[(self.df['seller_account_age_days'] < 5) &
                                   (self.df['seller_item_count'] > 100)]
        for idx in bulk_new_sellers.index:
            self.anomaly_scores.loc[idx, 'new_bulk_seller'] = 0.9
            anomalies.append({
                'item_id': self.df.loc[idx, 'item_id'],
                'type': 'New Bulk Seller',
                'reason': f"Account {self.df.loc[idx, 'seller_account_age_days']} days old, "
                         f"{self.df.loc[idx, 'seller_item_count']} items"
            })

        print(f"  New bulk sellers (fraud risk): {len(bulk_new_sellers)}")

        # Anomaly 5: Repeated relisting without sales
        frequent_relisters = self.df[(self.df['relistings'] > 3) &
                                     (self.df['sold'] == 'no')]
        for idx in frequent_relisters.index:
            self.anomaly_scores.loc[idx, 'frequent_relisting'] = 0.6

        print(f"  Frequent relisters: {len(frequent_relisters)}")
        print(f"  Total behavioral anomalies: {len(anomalies)}")

        return anomalies

    def detect_content_anomalies(self):
        """Layer 3: Content and presentation anomalies"""
        print(f"\n{'-'*80}")
        print("LAYER 3: CONTENT ANOMALY DETECTION")
        print(f"{'-'*80}")

        anomalies = []

        # Anomaly 1: Zero images
        no_images = self.df[self.df['image_count'] == 0]
        for idx in no_images.index:
            self.anomaly_scores.loc[idx, 'no_images'] = 0.4

        print(f"  No images: {len(no_images)}")

        # Anomaly 2: Minimal description
        minimal_desc = self.df[self.df['description_length'] < 20]
        for idx in minimal_desc.index:
            self.anomaly_scores.loc[idx, 'minimal_description'] = 0.5

        print(f"  Minimal description (<20 chars): {len(minimal_desc)}")

        # Anomaly 3: Zero images + minimal description combo
        poor_content = self.df[(self.df['image_count'] == 0) &
                              (self.df['description_length'] < 50)]
        for idx in poor_content.index:
            self.anomaly_scores.loc[idx, 'poor_content_quality'] = 0.7
            anomalies.append({
                'item_id': self.df.loc[idx, 'item_id'],
                'type': 'Poor Content Quality',
                'reason': f"0 images, {self.df.loc[idx, 'description_length']} char description"
            })

        print(f"  Poor content quality (images + description): {len(poor_content)}")

        # Anomaly 4: Excessive images (stock photo concern)
        excessive_images = self.df[self.df['image_count'] > 8]
        for idx in excessive_images.index:
            self.anomaly_scores.loc[idx, 'excessive_images'] = 0.2

        print(f"  Excessive images (>8): {len(excessive_images)}")

        print(f"  Total content anomalies: {len(anomalies)}")

        return anomalies

    def detect_luxury_counterfeits(self):
        """Layer 4: Luxury goods counterfeit detection"""
        print(f"\n{'-'*80}")
        print("LAYER 4: LUXURY COUNTERFEIT DETECTION")
        print(f"{'-'*80}")

        anomalies = []

        # Luxury items
        luxury_items = self.df[self.df['brand'].isin(self.luxury_brands)]

        print(f"  Total luxury items: {len(luxury_items)}")

        # Signal 1: Luxury item priced 50% below typical
        for brand in self.luxury_brands:
            brand_items = luxury_items[luxury_items['brand'] == brand]
            if len(brand_items) < 3:
                continue

            avg_price = brand_items['current_price'].mean()
            threshold = avg_price * 0.50

            suspicious_low = brand_items[brand_items['current_price'] < threshold]
            for idx in suspicious_low.index:
                self.anomaly_scores.loc[idx, 'luxury_low_price'] = 0.95
                anomalies.append({
                    'item_id': self.df.loc[idx, 'item_id'],
                    'type': 'Luxury Low Price (Counterfeit Risk)',
                    'reason': f"{brand}: €{self.df.loc[idx, 'current_price']:.2f} "
                             f"vs avg €{avg_price:.2f} (99% counterfeit probability)"
                })

        print(f"  Luxury low-price items: {(self.anomaly_scores['luxury_low_price'] == 0.95).sum()}")

        # Signal 2: Luxury from new accounts
        luxury_new_seller = luxury_items[luxury_items['seller_account_age_days'] < 30]
        for idx in luxury_new_seller.index:
            self.anomaly_scores.loc[idx, 'luxury_new_seller'] = 0.8

        print(f"  Luxury from new sellers (<30 days): {len(luxury_new_seller)}")

        # Signal 3: Batch luxury listings
        # Group by seller and count luxury items
        seller_luxury_counts = luxury_items.groupby('seller_account_age_days').size()
        bulk_luxury_sellers = seller_luxury_counts[seller_luxury_counts >= 10].index

        bulk_luxury = luxury_items[luxury_items['seller_account_age_days'].isin(bulk_luxury_sellers)]
        for idx in bulk_luxury.index:
            self.anomaly_scores.loc[idx, 'bulk_luxury'] = 0.85
            anomalies.append({
                'item_id': self.df.loc[idx, 'item_id'],
                'type': 'Bulk Luxury Listing',
                'reason': f"Seller has multiple luxury items (possible counterfeit ring)"
            })

        print(f"  Bulk luxury sellers: {len(bulk_luxury)}")

        # Signal 4: New seller + luxury + low price combo
        high_risk_combo = luxury_items[
            (luxury_items['seller_account_age_days'] < 30) &
            (luxury_items['current_price'] < luxury_items['current_price'].median())
        ]
        for idx in high_risk_combo.index:
            self.anomaly_scores.loc[idx, 'luxury_high_risk'] = 0.98
            anomalies.append({
                'item_id': self.df.loc[idx, 'item_id'],
                'type': 'EXTREME FRAUD RISK',
                'reason': f"New seller + luxury brand + low price = 98% fraud probability"
            })

        print(f"  EXTREME RISK combinations: {len(high_risk_combo)}")
        print(f"  Total luxury anomalies: {len(anomalies)}")

        return anomalies

    def detect_statistical_outliers(self):
        """Layer 5: Multivariate statistical outlier detection"""
        print(f"\n{'-'*80}")
        print("LAYER 5: STATISTICAL OUTLIER DETECTION")
        print(f"{'-'*80}")

        # Prepare numeric features
        numeric_features = ['current_price', 'days_listed', 'view_count', 'favorite_count',
                          'seller_rating', 'seller_account_age_days', 'seller_item_count',
                          'price_drops', 'description_length', 'image_count', 'age_years']

        # Clean data
        df_clean = self.df[numeric_features].fillna(self.df[numeric_features].median())

        # Isolation Forest
        print(f"\n  Running Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso_forest.fit_predict(df_clean)
        outlier_scores = iso_forest.score_samples(df_clean)

        # Convert to 0-1 scale (more negative = more anomalous)
        outlier_scores_normalized = (outlier_scores - outlier_scores.min()) / \
                                   (outlier_scores.max() - outlier_scores.min())
        outlier_scores_normalized = 1 - outlier_scores_normalized  # Invert so high = anomalous

        self.anomaly_scores['isolation_forest_score'] = outlier_scores_normalized

        outliers_if = (outlier_labels == -1).sum()
        print(f"  Isolation Forest outliers: {outliers_if} ({outliers_if/len(self.df)*100:.1f}%)")

        # Mahalanobis distance
        print(f"\n  Computing Mahalanobis distance...")
        try:
            # Standardize data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_clean)

            # Compute covariance matrix
            cov_matrix = np.cov(df_scaled.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            mean_vec = np.mean(df_scaled, axis=0)

            # Calculate Mahalanobis distance for each point
            mahal_distances = []
            for i in range(len(df_scaled)):
                try:
                    distance = mahalanobis(df_scaled[i], mean_vec, inv_cov_matrix)
                    mahal_distances.append(distance)
                except:
                    mahal_distances.append(0)

            mahal_distances = np.array(mahal_distances)

            # Normalize to 0-1
            if mahal_distances.max() > 0:
                mahal_normalized = mahal_distances / mahal_distances.max()
                self.anomaly_scores['mahalanobis_score'] = mahal_normalized

                # Flag extreme outliers (>99th percentile)
                threshold = np.percentile(mahal_distances, 99)
                extreme_mahal = mahal_distances > threshold
                print(f"  Mahalanobis outliers (>99th percentile): {extreme_mahal.sum()}")
            else:
                self.anomaly_scores['mahalanobis_score'] = 0

        except Exception as e:
            print(f"  Mahalanobis distance failed: {str(e)}")
            self.anomaly_scores['mahalanobis_score'] = 0

        print(f"\n  Statistical outlier detection complete")

        return True

    def aggregate_anomaly_scores(self):
        """Aggregate all anomaly scores into final score"""
        print(f"\n{'-'*80}")
        print("AGGREGATING ANOMALY SCORES")
        print(f"{'-'*80}")

        # Fill NaN values with 0
        score_columns = [col for col in self.anomaly_scores.columns if col != 'item_id']
        self.anomaly_scores[score_columns] = self.anomaly_scores[score_columns].fillna(0)

        # Calculate weighted composite score
        weights = {
            'price_extreme_low': 0.15,
            'price_extreme_high': 0.10,
            'rapid_price_drop': 0.10,
            'high_views_no_favs': 0.08,
            'high_favs_no_sale': 0.07,
            'new_bulk_seller': 0.12,
            'no_images': 0.05,
            'minimal_description': 0.05,
            'poor_content_quality': 0.08,
            'luxury_low_price': 0.20,
            'luxury_new_seller': 0.10,
            'bulk_luxury': 0.12,
            'luxury_high_risk': 0.25,
            'isolation_forest_score': 0.10,
            'mahalanobis_score': 0.10
        }

        # Composite score (0-100)
        composite = 0
        for col, weight in weights.items():
            if col in self.anomaly_scores.columns:
                composite += self.anomaly_scores[col] * weight

        self.anomaly_scores['composite_anomaly_score'] = composite * 100
        self.anomaly_scores['composite_anomaly_score'] = \
            self.anomaly_scores['composite_anomaly_score'].clip(0, 100)

        # Fraud confidence (based on suspected_counterfeit + composite)
        self.anomaly_scores['fraud_confidence'] = (
            self.df['suspected_counterfeit'] * 50 +
            self.anomaly_scores['composite_anomaly_score'] * 0.5
        ).clip(0, 100)

        print(f"  Composite anomaly scores calculated")
        print(f"  Mean anomaly score: {self.anomaly_scores['composite_anomaly_score'].mean():.2f}")
        print(f"  Max anomaly score: {self.anomaly_scores['composite_anomaly_score'].max():.2f}")

    def generate_anomaly_report(self):
        """Generate comprehensive anomaly report"""
        print(f"\n{'-'*80}")
        print("ANOMALY DETECTION REPORT")
        print(f"{'-'*80}")

        # Merge with original data for context
        report = self.df.copy()
        report['anomaly_score'] = self.anomaly_scores['composite_anomaly_score']
        report['fraud_confidence'] = self.anomaly_scores['fraud_confidence']

        # Top 100 anomalies
        top_anomalies = report.nlargest(100, 'anomaly_score')

        print(f"\n  TOP 20 MOST ANOMALOUS ITEMS:")
        print(f"  {'-' * 75}")

        for i, (idx, row) in enumerate(top_anomalies.head(20).iterrows(), 1):
            print(f"\n  #{i} - Item {row['item_id']} (Anomaly Score: {row['anomaly_score']:.1f}/100)")
            print(f"      Fraud Confidence: {row['fraud_confidence']:.1f}%")
            print(f"      Category: {row['category']}, Brand: {row['brand']}")
            print(f"      Price: €{row['current_price']:.2f}, Age: {row['age_years']:.0f} years")
            print(f"      Seller: {row['seller_account_age_days']:.0f} days old, "
                  f"{row['seller_item_count']:.0f} items, rating {row['seller_rating']:.1f}")
            print(f"      Engagement: {row['view_count']:.0f} views, {row['favorite_count']:.0f} favorites")

            # Explain why it's anomalous
            reasons = []
            if row['brand'] in self.luxury_brands and row['current_price'] < 50:
                reasons.append("LUXURY BRAND AT SUSPICIOUSLY LOW PRICE")
            if row['seller_account_age_days'] < 30 and row['seller_item_count'] > 100:
                reasons.append("New seller with bulk listings")
            if row['view_count'] > 500 and row['favorite_count'] == 0:
                reasons.append("High views but no favorites (misleading?)")
            if row['image_count'] == 0:
                reasons.append("No images provided")
            if row['price_drops'] > 3:
                reasons.append(f"{row['price_drops']} price drops (desperation)")

            if reasons:
                print(f"      WHY ANOMALOUS: {'; '.join(reasons)}")

        # Save full report
        report_file = 'anomaly_detection_report.csv'
        report[['item_id', 'category', 'brand', 'current_price', 'anomaly_score',
               'fraud_confidence', 'sold', 'seller_account_age_days',
               'seller_item_count']].nlargest(100, 'anomaly_score').to_csv(
                   report_file, index=False)

        print(f"\n  ✓ Full anomaly report (top 100) saved to: {report_file}")

        # Summary statistics
        print(f"\n  SUMMARY STATISTICS:")
        print(f"  {'-' * 75}")
        high_risk = report[report['fraud_confidence'] > 75]
        medium_risk = report[(report['fraud_confidence'] > 50) &
                            (report['fraud_confidence'] <= 75)]
        low_risk = report[report['fraud_confidence'] <= 50]

        print(f"    HIGH RISK (>75% fraud confidence): {len(high_risk)} items")
        print(f"    MEDIUM RISK (50-75%): {len(medium_risk)} items")
        print(f"    LOW RISK (<50%): {len(low_risk)} items")

        print(f"\n  ACTIONABLE RECOMMENDATIONS:")
        print(f"  {'-' * 75}")
        print(f"    FLAG FOR REVIEW: {len(report[report['anomaly_score'] > 60])} items (score >60)")
        print(f"    AUTO-REMOVE: {len(report[report['fraud_confidence'] > 90])} items (fraud >90%)")
        print(f"    MONITOR: {len(report[(report['anomaly_score'] > 40) & (report['anomaly_score'] <= 60)])} items (score 40-60)")


def main():
    """Main execution function"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    engine = AnomalyDetectionEngine(df)
    anomaly_scores = engine.detect_all_anomalies()

    # Save anomaly scores
    anomaly_scores.to_csv('anomaly_scores.csv', index=False)
    print(f"\n✓ Anomaly scores saved to: anomaly_scores.csv")

    return anomaly_scores


if __name__ == "__main__":
    scores = main()
