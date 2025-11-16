"""
PHASE 6: PATTERN DISCOVERY & CLUSTERING
K-Means clustering, hierarchical clustering, and association rule mining
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class PatternDiscoveryEngine:
    """Discovers patterns through clustering and association rules"""

    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()

    def discover_all_patterns(self):
        """Run all pattern discovery methods"""
        print("\n" + "="*80)
        print("PHASE 6: PATTERN DISCOVERY & CLUSTERING")
        print("="*80)

        # K-Means clustering
        self.kmeans_clustering()

        # Hierarchical clustering
        self.hierarchical_clustering()

        # DBSCAN (density-based)
        self.dbscan_clustering()

        # Association rules
        self.association_rule_mining()

        print("\n" + "="*80)
        print("PHASE 6 COMPLETE")
        print("="*80)

    def prepare_features(self):
        """Prepare features for clustering"""
        # Select numeric features
        feature_cols = ['current_price', 'days_listed', 'view_count', 'favorite_count',
                       'seller_rating', 'seller_account_age_days', 'seller_item_count',
                       'price_drops', 'description_length', 'image_count', 'age_years',
                       'suspected_counterfeit']

        # Encode categorical features
        le_category = LabelEncoder()
        le_platform = LabelEncoder()
        le_condition = LabelEncoder()

        df_features = self.df[feature_cols].copy()
        df_features['category_encoded'] = le_category.fit_transform(self.df['category'])
        df_features['platform_encoded'] = le_platform.fit_transform(self.df['platform'])
        df_features['condition_encoded'] = le_condition.fit_transform(self.df['condition'])
        df_features['sold_encoded'] = (self.df['sold'] == 'yes').astype(int)

        # Handle missing values
        df_features = df_features.fillna(df_features.median())

        # Standardize features
        features_scaled = self.scaler.fit_transform(df_features)

        return features_scaled, df_features.columns

    def kmeans_clustering(self):
        """K-Means clustering to find natural buyer/seller profiles"""
        print(f"\n{'-'*80}")
        print("K-MEANS CLUSTERING ANALYSIS")
        print(f"{'-'*80}")

        features_scaled, feature_names = self.prepare_features()

        # Determine optimal k using elbow method and silhouette score
        print("\n  Finding optimal number of clusters...")
        silhouette_scores = []
        K_range = range(3, 9)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            silhouette_scores.append(score)
            print(f"    k={k}: silhouette score = {score:.3f}")

        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\n  Optimal k: {optimal_k} (highest silhouette score: {max(silhouette_scores):.3f})")

        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        self.df['cluster'] = cluster_labels

        # Analyze each cluster
        print(f"\n  CLUSTER PROFILES:")
        print(f"  {'-' * 75}")

        cluster_names = []
        for i in range(optimal_k):
            cluster_data = self.df[self.df['cluster'] == i]
            print(f"\n  CLUSTER {i} ({len(cluster_data)} items, {len(cluster_data)/len(self.df)*100:.1f}%)")
            print(f"  {'-' * 70}")

            # Characterize cluster
            avg_price = cluster_data['current_price'].mean()
            avg_seller_items = cluster_data['seller_item_count'].mean()
            avg_rating = cluster_data['seller_rating'].mean()
            sale_rate = (cluster_data['sold'] == 'yes').sum() / len(cluster_data) * 100
            fraud_rate = cluster_data['fraud_flag'].sum() / len(cluster_data) * 100

            print(f"    Avg price: €{avg_price:.2f}")
            print(f"    Avg seller items: {avg_seller_items:.0f}")
            print(f"    Avg seller rating: {avg_rating:.2f}/5.0")
            print(f"    Sale success rate: {sale_rate:.1f}%")
            print(f"    Fraud flag rate: {fraud_rate:.1f}%")

            # Most common category and platform
            top_category = cluster_data['category'].mode()[0] if len(cluster_data) > 0 else 'N/A'
            top_platform = cluster_data['platform'].mode()[0] if len(cluster_data) > 0 else 'N/A'
            print(f"    Most common category: {top_category}")
            print(f"    Most common platform: {top_platform}")

            # Cluster interpretation
            if avg_seller_items > 500 and avg_rating > 4.0 and fraud_rate < 2:
                cluster_name = "Professional Sellers (High Volume, Consistent)"
            elif avg_seller_items < 50 and avg_price < 100:
                cluster_name = "Casual Sellers (One-off Items)"
            elif avg_price > 500 or fraud_rate > 10:
                cluster_name = "Luxury/High-Risk Items"
            elif sale_rate < 20:
                cluster_name = "Struggling Sellers (Low Success)"
            elif fraud_rate > 15:
                cluster_name = "Potential Fraud Cluster"
            else:
                cluster_name = f"Mixed Profile {i}"

            cluster_names.append(cluster_name)
            print(f"    → INTERPRETATION: {cluster_name}")

        # Save clustering results
        cluster_summary = self.df.groupby('cluster').agg({
            'item_id': 'count',
            'current_price': 'mean',
            'sold': lambda x: (x == 'yes').sum() / len(x) * 100,
            'fraud_flag': lambda x: x.sum() / len(x) * 100,
            'seller_item_count': 'mean',
            'seller_rating': 'mean'
        }).round(2)
        cluster_summary.columns = ['Count', 'Avg_Price', 'Sale_Rate_%', 'Fraud_Rate_%',
                                   'Avg_Seller_Items', 'Avg_Rating']
        cluster_summary['Cluster_Name'] = cluster_names

        print(f"\n  CLUSTER COMPARISON TABLE:")
        print(f"  {'-' * 75}")
        print(cluster_summary.to_string())

        cluster_summary.to_csv('cluster_analysis.csv')
        print(f"\n  ✓ Cluster analysis saved to: cluster_analysis.csv")

    def hierarchical_clustering(self):
        """Hierarchical clustering with dendrogram interpretation"""
        print(f"\n{'-'*80}")
        print("HIERARCHICAL CLUSTERING ANALYSIS")
        print(f"{'-'*80}")

        # Sample for performance (full dataset too large for dendrogram)
        sample_size = min(1000, len(self.df))
        sample_indices = np.random.choice(self.df.index, sample_size, replace=False)
        df_sample = self.df.loc[sample_indices]

        # Prepare features for sample
        feature_cols = ['current_price', 'seller_item_count', 'seller_rating',
                       'suspected_counterfeit', 'days_listed']
        X_sample = df_sample[feature_cols].fillna(df_sample[feature_cols].median())
        X_sample_scaled = self.scaler.fit_transform(X_sample)

        # Perform hierarchical clustering
        print(f"\n  Performing hierarchical clustering on {sample_size} items...")
        linkage_matrix = linkage(X_sample_scaled, method='ward')

        # Identify major clusters at different cut heights
        agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
        hierarchical_labels = agg_clustering.fit_predict(X_sample_scaled)

        print(f"\n  HIERARCHICAL STRUCTURE (5 major clusters):")
        print(f"  {'-' * 75}")

        for i in range(5):
            cluster_items = df_sample[hierarchical_labels == i]
            print(f"\n  Super-Cluster {i}: {len(cluster_items)} items")
            print(f"    Avg price: €{cluster_items['current_price'].mean():.2f}")
            print(f"    Avg fraud score: {cluster_items['suspected_counterfeit'].mean():.3f}")
            print(f"    Sale rate: {(cluster_items['sold'] == 'yes').mean()*100:.1f}%")

        print(f"\n  → Hierarchical clustering reveals nested marketplace segments")
        print(f"  → Some sellers naturally group into super-categories (e.g., professional vs casual)")

    def dbscan_clustering(self):
        """DBSCAN density-based clustering"""
        print(f"\n{'-'*80}")
        print("DBSCAN DENSITY-BASED CLUSTERING")
        print(f"{'-'*80}")

        features_scaled, _ = self.prepare_features()

        # Run DBSCAN
        print(f"\n  Running DBSCAN...")
        dbscan = DBSCAN(eps=3.0, min_samples=50)
        dbscan_labels = dbscan.fit_predict(features_scaled)

        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)

        print(f"    Clusters found: {n_clusters}")
        print(f"    Noise points (outliers): {n_noise} ({n_noise/len(self.df)*100:.1f}%)")

        # Analyze dense regions vs noise
        print(f"\n  DENSE REGIONS (Normal Activity):")
        for i in range(min(3, n_clusters)):
            if i in dbscan_labels:
                cluster_items = self.df[dbscan_labels == i]
                print(f"    Dense cluster {i}: {len(cluster_items)} items")
                print(f"      Avg fraud score: {cluster_items['suspected_counterfeit'].mean():.3f}")
                print(f"      → Represents mainstream marketplace patterns")

        # Noise points analysis
        if n_noise > 0:
            noise_items = self.df[dbscan_labels == -1]
            print(f"\n  NOISE POINTS (Anomalies):")
            print(f"    {n_noise} items don't fit any dense cluster")
            print(f"    Avg fraud score: {noise_items['suspected_counterfeit'].mean():.3f}")
            print(f"    → These are true outliers: rare, unique, or suspicious")
            print(f"    → {(noise_items['fraud_flag'] == True).sum()} are fraud-flagged "
                  f"({(noise_items['fraud_flag'] == True).sum()/len(noise_items)*100:.1f}%)")

    def association_rule_mining(self):
        """Mine association rules (market basket analysis)"""
        print(f"\n{'-'*80}")
        print("ASSOCIATION RULE MINING")
        print(f"{'-'*80}")

        # Create binary features for rule mining
        rules = []

        # Rule 1: Electronics + high price drop → sold
        electronics_drops = self.df[(self.df['category'] == 'electronics') &
                                    (self.df['price_drops'] > 2)]
        if len(electronics_drops) > 0:
            confidence = (electronics_drops['sold'] == 'yes').sum() / len(electronics_drops)
            support = len(electronics_drops) / len(self.df)
            rules.append({
                'Rule': 'IF (electronics AND price_drops>2) THEN sold',
                'Support': support,
                'Confidence': confidence,
                'Interpretation': 'Electronics with price corrections tend to sell'
            })
            print(f"\n  Rule 1:")
            print(f"    IF (category=electronics AND price_drops>2) THEN sold=yes")
            print(f"    Support: {support:.3f}, Confidence: {confidence:.3f} ({confidence*100:.1f}%)")

        # Rule 2: New seller + luxury + short description → fraud
        risky_combo = self.df[(self.df['seller_account_age_days'] < 30) &
                             (self.df['description_length'] < 50) &
                             (self.df['suspected_counterfeit'] > 0.5)]
        if len(risky_combo) > 0:
            confidence = risky_combo['fraud_flag'].sum() / len(risky_combo)
            support = len(risky_combo) / len(self.df)
            rules.append({
                'Rule': 'IF (new_seller AND short_desc AND high_counterfeit) THEN fraud',
                'Support': support,
                'Confidence': confidence,
                'Interpretation': 'New sellers with minimal descriptions are high fraud risk'
            })
            print(f"\n  Rule 2:")
            print(f"    IF (seller_new AND description<50 AND suspected_counterfeit>0.5) THEN fraud")
            print(f"    Support: {support:.3f}, Confidence: {confidence:.3f} ({confidence*100:.1f}%)")

        # Rule 3: Facebook + multiple relistings → unsold
        fb_relisters = self.df[(self.df['platform'] == 'Facebook Marketplace') &
                              (self.df['relistings'] > 2)]
        if len(fb_relisters) > 0:
            confidence = (fb_relisters['sold'] == 'no').sum() / len(fb_relisters)
            support = len(fb_relisters) / len(self.df)
            rules.append({
                'Rule': 'IF (Facebook AND relistings>2) THEN unsold',
                'Support': support,
                'Confidence': confidence,
                'Interpretation': 'Multiple relistings on Facebook often fail to sell'
            })
            print(f"\n  Rule 3:")
            print(f"    IF (platform=Facebook AND relistings>2) THEN sold=no")
            print(f"    Support: {support:.3f}, Confidence: {confidence:.3f} ({confidence*100:.1f}%)")

        # Rule 4: High views + low favorites → misleading
        misleading = self.df[(self.df['view_count'] > 500) &
                            (self.df['favorite_count'] < 10)]
        if len(misleading) > 0:
            print(f"\n  Rule 4:")
            print(f"    IF (view_count>500 AND favorite_count<10) THEN misleading_listing")
            print(f"    Support: {len(misleading)/len(self.df):.3f}")
            print(f"    → {len(misleading)} items attract views but no interest (clickbait?)")

        # Rule 5: Age > 15 + collectible → premium price
        vintage_collectibles = self.df[(self.df['age_years'] > 15) &
                                       (self.df['category'] == 'collectibles')]
        if len(vintage_collectibles) > 0:
            premium_items = vintage_collectibles[
                vintage_collectibles['current_price'] > vintage_collectibles['current_price'].median()
            ]
            confidence = len(premium_items) / len(vintage_collectibles)
            print(f"\n  Rule 5:")
            print(f"    IF (age_years>15 AND category=collectible) THEN premium_price")
            print(f"    Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            print(f"    → Vintage collectibles often command premium prices")

        # Rule 6: No images + high rating → exception case
        no_img_good_seller = self.df[(self.df['image_count'] == 0) &
                                     (self.df['seller_rating'] > 4.5)]
        if len(no_img_good_seller) > 0:
            sale_rate = (no_img_good_seller['sold'] == 'yes').mean()
            print(f"\n  Rule 6:")
            print(f"    IF (image_count=0 AND seller_rating>4.5) THEN ???")
            print(f"    Support: {len(no_img_good_seller)/len(self.df):.3f}")
            print(f"    Sale rate: {sale_rate*100:.1f}%")
            print(f"    → Interesting exception: Reputable sellers succeed without images")

        # Summary
        print(f"\n  ASSOCIATION RULES SUMMARY:")
        print(f"  {'-' * 75}")
        if len(rules) > 0:
            rules_df = pd.DataFrame(rules)
            print(rules_df.to_string(index=False))

            rules_df.to_csv('association_rules.csv', index=False)
            print(f"\n  ✓ Association rules saved to: association_rules.csv")


def main():
    """Main execution function"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    engine = PatternDiscoveryEngine(df)
    engine.discover_all_patterns()

    print(f"\n✓ Pattern discovery complete")

    return engine


if __name__ == "__main__":
    engine = main()
