"""
INTERACTIVE VISUALIZATION DASHBOARD
Creates comprehensive interactive visualizations using Plotly
All charts are HTML-based and can be opened in browser
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class VisualizationDashboard:
    """Creates interactive visualizations for marketplace analysis"""

    def __init__(self, data_file='marketplace_data.csv',
                 anomaly_file='anomaly_scores.csv'):
        self.df = pd.read_csv(data_file)
        self.anomaly_scores = pd.read_csv(anomaly_file)

        # Merge anomaly scores
        self.df = self.df.merge(
            self.anomaly_scores[['item_id', 'composite_anomaly_score', 'fraud_confidence']],
            on='item_id',
            how='left'
        )

    def create_all_visualizations(self):
        """Generate all visualizations and save as HTML"""
        print("\n" + "="*80)
        print("CREATING INTERACTIVE VISUALIZATION DASHBOARD")
        print("="*80)

        # 1. Price distribution heatmap
        self.price_distribution_heatmap()

        # 2. Fraud risk scatter plot
        self.fraud_risk_scatter()

        # 3. Cluster visualization
        self.cluster_visualization()

        # 4. Time series patterns
        self.seasonal_patterns()

        # 5. Anomaly score distributions
        self.anomaly_distributions()

        # 6. Platform comparison
        self.platform_comparison()

        # 7. Seller journey heatmap
        self.seller_journey()

        # 8. Correlation heatmap
        self.correlation_heatmap()

        # 9. Combined dashboard
        self.create_combined_dashboard()

        print("\n✓ All visualizations created!")
        print("\nGenerated files:")
        print("  - viz_price_heatmap.html")
        print("  - viz_fraud_scatter.html")
        print("  - viz_clusters.html")
        print("  - viz_seasonal.html")
        print("  - viz_anomaly_dist.html")
        print("  - viz_platform_compare.html")
        print("  - viz_seller_journey.html")
        print("  - viz_correlation.html")
        print("  - DASHBOARD.html (combined)")

    def price_distribution_heatmap(self):
        """Price distribution heatmap by category and condition"""
        print("\n  Creating price distribution heatmap...")

        pivot_data = self.df.pivot_table(
            values='current_price',
            index='category',
            columns='condition',
            aggfunc='mean'
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='YlOrRd',
            text=np.round(pivot_data.values, 2),
            texttemplate='€%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Avg Price (€)")
        ))

        fig.update_layout(
            title='Average Price Heatmap: Category × Condition',
            xaxis_title='Condition',
            yaxis_title='Category',
            width=900,
            height=600,
            font=dict(size=14)
        )

        fig.write_html('viz_price_heatmap.html')
        print("    ✓ Saved: viz_price_heatmap.html")

    def fraud_risk_scatter(self):
        """3D scatter plot: seller age vs items vs fraud risk"""
        print("\n  Creating fraud risk scatter plot...")

        # Sample for performance (plot 2000 points)
        sample = self.df.sample(min(2000, len(self.df)))

        # Remove rows with NaN in critical columns
        sample = sample.dropna(subset=['seller_account_age_days', 'seller_item_count',
                                       'fraud_confidence', 'current_price'])

        fig = px.scatter_3d(
            sample,
            x='seller_account_age_days',
            y='seller_item_count',
            z='fraud_confidence',
            color='fraud_confidence',
            size='current_price',
            hover_data=['category', 'brand', 'platform'],
            color_continuous_scale='Reds',
            labels={
                'seller_account_age_days': 'Seller Age (days)',
                'seller_item_count': 'Seller Items',
                'fraud_confidence': 'Fraud Risk %'
            },
            title='Fraud Risk Analysis: Seller Profile vs Risk Score'
        )

        fig.update_layout(
            width=1000,
            height=700,
            scene=dict(
                xaxis_title='Seller Age (days)',
                yaxis_title='Items Listed',
                zaxis_title='Fraud Risk %'
            )
        )

        fig.write_html('viz_fraud_scatter.html')
        print("    ✓ Saved: viz_fraud_scatter.html")

    def cluster_visualization(self):
        """2D PCA visualization of clusters"""
        print("\n  Creating cluster visualization...")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Prepare features for PCA
        feature_cols = ['current_price', 'seller_item_count', 'seller_rating',
                       'view_count', 'favorite_count', 'days_listed']

        df_clean = self.df[feature_cols].fillna(self.df[feature_cols].median())

        # Standardize and apply PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Add to dataframe
        plot_df = self.df.copy()
        plot_df['PC1'] = pca_result[:, 0]
        plot_df['PC2'] = pca_result[:, 1]

        # Sample for performance
        plot_df = plot_df.sample(min(3000, len(plot_df)))

        fig = px.scatter(
            plot_df,
            x='PC1',
            y='PC2',
            color='category',
            size='current_price',
            hover_data=['brand', 'platform', 'sold'],
            title=f'Marketplace Clusters (PCA): Explained Variance = {sum(pca.explained_variance_ratio_)*100:.1f}%',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
            }
        )

        fig.update_layout(width=1000, height=700)
        fig.write_html('viz_clusters.html')
        print("    ✓ Saved: viz_clusters.html")

    def seasonal_patterns(self):
        """Time series: monthly listing patterns by category"""
        print("\n  Creating seasonal pattern charts...")

        monthly_counts = self.df.groupby(['month_listed', 'category']).size().reset_index(name='count')

        fig = px.line(
            monthly_counts,
            x='month_listed',
            y='count',
            color='category',
            title='Seasonal Listing Patterns by Category',
            labels={
                'month_listed': 'Month',
                'count': 'Number of Listings',
                'category': 'Category'
            },
            markers=True
        )

        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )

        fig.update_layout(width=1000, height=600)
        fig.write_html('viz_seasonal.html')
        print("    ✓ Saved: viz_seasonal.html")

    def anomaly_distributions(self):
        """Distribution of anomaly scores"""
        print("\n  Creating anomaly distribution charts...")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Anomaly Score Distribution',
                          'Fraud Confidence Distribution',
                          'High-Risk Items by Category',
                          'High-Risk Items by Platform')
        )

        # 1. Anomaly score histogram
        fig.add_trace(
            go.Histogram(x=self.df['composite_anomaly_score'],
                        name='Anomaly Score',
                        marker_color='indianred',
                        nbinsx=50),
            row=1, col=1
        )

        # 2. Fraud confidence histogram
        fig.add_trace(
            go.Histogram(x=self.df['fraud_confidence'],
                        name='Fraud Confidence',
                        marker_color='darkred',
                        nbinsx=50),
            row=1, col=2
        )

        # 3. High-risk by category
        high_risk = self.df[self.df['fraud_confidence'] > 50]
        category_counts = high_risk['category'].value_counts()

        fig.add_trace(
            go.Bar(x=category_counts.index, y=category_counts.values,
                  marker_color='crimson',
                  name='High Risk Count'),
            row=2, col=1
        )

        # 4. High-risk by platform
        platform_counts = high_risk['platform'].value_counts()

        fig.add_trace(
            go.Bar(x=platform_counts.index, y=platform_counts.values,
                  marker_color='firebrick',
                  name='High Risk Count'),
            row=2, col=2
        )

        fig.update_layout(
            title_text='Anomaly & Fraud Analysis Dashboard',
            showlegend=False,
            height=800,
            width=1200
        )

        fig.write_html('viz_anomaly_dist.html')
        print("    ✓ Saved: viz_anomaly_dist.html")

    def platform_comparison(self):
        """Platform comparison radar chart"""
        print("\n  Creating platform comparison chart...")

        platforms = self.df['platform'].unique()

        metrics = []
        for platform in platforms:
            pdata = self.df[self.df['platform'] == platform]
            metrics.append({
                'Platform': platform,
                'Sale Rate': (pdata['sold'] == 'yes').mean() * 100,
                'Avg Price': pdata['current_price'].mean(),
                'Avg Rating': pdata['seller_rating'].mean(),
                'Fraud Risk': pdata['fraud_confidence'].mean()
            })

        metrics_df = pd.DataFrame(metrics)

        # Normalize metrics for radar chart (0-100 scale)
        normalized = metrics_df.copy()
        normalized['Sale Rate'] = metrics_df['Sale Rate']  # Already 0-100
        normalized['Avg Price'] = (metrics_df['Avg Price'] / metrics_df['Avg Price'].max()) * 100
        normalized['Avg Rating'] = (metrics_df['Avg Rating'] / 5.0) * 100
        normalized['Fraud Risk'] = 100 - metrics_df['Fraud Risk']  # Invert (higher = better)

        fig = go.Figure()

        for idx, row in normalized.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Sale Rate'], row['Avg Price'], row['Avg Rating'], row['Fraud Risk']],
                theta=['Sale Rate', 'Price Level', 'Seller Rating', 'Safety'],
                fill='toself',
                name=row['Platform']
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title='Platform Performance Comparison (Normalized)',
            width=800,
            height=600
        )

        fig.write_html('viz_platform_compare.html')
        print("    ✓ Saved: viz_platform_compare.html")

    def seller_journey(self):
        """Seller experience journey heatmap"""
        print("\n  Creating seller journey heatmap...")

        # Create experience bins
        self.df['experience_bin'] = pd.cut(
            self.df['seller_account_age_days'],
            bins=[0, 90, 365, 730, 10000],
            labels=['New (0-3mo)', 'Developing (3-12mo)',
                   'Experienced (1-2y)', 'Veteran (2y+)']
        )

        # Create item volume bins
        self.df['volume_bin'] = pd.cut(
            self.df['seller_item_count'],
            bins=[0, 10, 50, 100, 10000],
            labels=['Casual (1-10)', 'Active (10-50)',
                   'Professional (50-100)', 'Power (100+)']
        )

        # Pivot: sale rate by experience and volume
        pivot = self.df.groupby(['experience_bin', 'volume_bin']).apply(
            lambda x: (x['sold'] == 'yes').mean() * 100
        ).unstack()

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 12},
            colorbar=dict(title="Sale Rate %")
        ))

        fig.update_layout(
            title='Seller Journey: Sale Success by Experience × Volume',
            xaxis_title='Listing Volume',
            yaxis_title='Seller Experience',
            width=900,
            height=600
        )

        fig.write_html('viz_seller_journey.html')
        print("    ✓ Saved: viz_seller_journey.html")

    def correlation_heatmap(self):
        """Interactive correlation heatmap"""
        print("\n  Creating correlation heatmap...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title='Feature Correlation Matrix',
            width=1200,
            height=1000,
            xaxis={'side': 'bottom'},
        )

        fig.update_xaxes(tickangle=45)

        fig.write_html('viz_correlation.html')
        print("    ✓ Saved: viz_correlation.html")

    def create_combined_dashboard(self):
        """Create single HTML page with all key visualizations"""
        print("\n  Creating combined dashboard...")

        # Simple HTML template combining all visualizations
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Marketplace Fraud Detection Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; margin-top: 40px; }}
                .viz-container {{ background: white; padding: 20px; margin: 20px 0;
                                 border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .summary {{ background: #e3f2fd; padding: 15px; border-radius: 8px;
                           margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #1976d2; }}
                .metric-label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>🔍 Marketplace Fraud Detection & Analytics Dashboard</h1>

            <div class="summary">
                <h2>Key Metrics</h2>
                <div class="metric">
                    <div class="metric-value">{total_items:,}</div>
                    <div class="metric-label">Total Items</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{fraud_count}</div>
                    <div class="metric-label">High-Risk Items</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{sale_rate:.1f}%</div>
                    <div class="metric-label">Sale Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">€{avg_price:.2f}</div>
                    <div class="metric-label">Avg Price</div>
                </div>
            </div>

            <div class="viz-container">
                <h2>📊 All Visualizations</h2>
                <p>Open individual files for interactive charts:</p>
                <ul>
                    <li><a href="viz_price_heatmap.html">Price Distribution Heatmap</a></li>
                    <li><a href="viz_fraud_scatter.html">Fraud Risk 3D Scatter</a></li>
                    <li><a href="viz_clusters.html">Cluster Visualization (PCA)</a></li>
                    <li><a href="viz_seasonal.html">Seasonal Patterns</a></li>
                    <li><a href="viz_anomaly_dist.html">Anomaly Distributions</a></li>
                    <li><a href="viz_platform_compare.html">Platform Comparison Radar</a></li>
                    <li><a href="viz_seller_journey.html">Seller Journey Heatmap</a></li>
                    <li><a href="viz_correlation.html">Correlation Matrix</a></li>
                </ul>
            </div>

            <div class="viz-container">
                <iframe src="viz_fraud_scatter.html" width="100%" height="750" frameborder="0"></iframe>
            </div>

            <div class="viz-container">
                <iframe src="viz_price_heatmap.html" width="100%" height="650" frameborder="0"></iframe>
            </div>

            <div class="viz-container">
                <iframe src="viz_seasonal.html" width="100%" height="650" frameborder="0"></iframe>
            </div>
        </body>
        </html>
        """

        # Calculate metrics
        metrics = {
            'total_items': len(self.df),
            'fraud_count': (self.df['fraud_confidence'] > 75).sum(),
            'sale_rate': (self.df['sold'] == 'yes').mean() * 100,
            'avg_price': self.df['current_price'].mean()
        }

        html_output = html_template.format(**metrics)

        with open('DASHBOARD.html', 'w') as f:
            f.write(html_output)

        print("    ✓ Saved: DASHBOARD.html")


def main():
    """Main execution"""
    print("\nLoading data for visualization...")

    dashboard = VisualizationDashboard()
    dashboard.create_all_visualizations()

    print("\n" + "="*80)
    print("VISUALIZATION DASHBOARD COMPLETE")
    print("="*80)
    print("\n📊 Open DASHBOARD.html in your browser to view all visualizations!")
    print("   All charts are interactive (zoom, pan, hover for details)")


if __name__ == "__main__":
    main()
