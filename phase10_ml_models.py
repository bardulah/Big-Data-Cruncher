"""
PHASE 10: ADVANCED ML MODELS
Train Random Forest, Gradient Boosting, and ensemble models for fraud prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, precision_recall_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')


class MLFraudDetector:
    """Advanced ML models for fraud detection"""

    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def train_all_models(self):
        """Train all ML models"""
        print("\n" + "="*80)
        print("PHASE 10: ADVANCED ML FRAUD DETECTION MODELS")
        print("="*80)

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data()

        # Train models
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_gradient_boosting(X_train, X_test, y_train, y_test)
        self.train_logistic_regression(X_train, X_test, y_train, y_test)

        # Ensemble model
        self.create_ensemble_model(X_test, y_test)

        # Feature importance
        self.analyze_feature_importance()

        # Save models
        self.save_models()

        print("\n" + "="*80)
        print("PHASE 10 COMPLETE")
        print("="*80)

    def prepare_training_data(self):
        """Prepare features and labels for training"""
        print(f"\n{'-'*80}")
        print("PREPARING TRAINING DATA")
        print(f"{'-'*80}")

        # Create binary fraud label
        # Use suspected_counterfeit > 0.5 as fraud
        self.df['is_fraud'] = (self.df['suspected_counterfeit'] > 0.5).astype(int)

        print(f"\n  Fraud distribution:")
        print(f"    Legitimate items: {(self.df['is_fraud'] == 0).sum():,} ({(self.df['is_fraud'] == 0).sum()/len(self.df)*100:.1f}%)")
        print(f"    Fraudulent items: {(self.df['is_fraud'] == 1).sum():,} ({(self.df['is_fraud'] == 1).sum()/len(self.df)*100:.1f}%)")

        # Select features
        feature_cols = [
            'current_price', 'original_price', 'age_years',
            'seller_account_age_days', 'seller_item_count', 'seller_rating',
            'image_count', 'description_length', 'days_listed',
            'view_count', 'favorite_count', 'price_drops', 'relistings'
        ]

        # Encode categorical features
        categorical_features = ['category', 'platform', 'condition', 'brand']

        for col in categorical_features:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
            self.label_encoders[col] = le
            feature_cols.append(f'{col}_encoded')

        # Add binary features
        self.df['shipping_offered_int'] = self.df['shipping_offered'].astype(int)
        feature_cols.append('shipping_offered_int')

        # Handle missing values
        X = self.df[feature_cols].fillna(self.df[feature_cols].median())
        y = self.df['is_fraud']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

        print(f"\n  Training set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        print(f"  Features: {len(feature_cols)}")

        self.feature_cols = feature_cols

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest classifier"""
        print(f"\n{'-'*80}")
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print(f"{'-'*80}")

        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        # Evaluate
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)

        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

        print(f"\n  Model Performance:")
        print(f"    Training accuracy: {train_score:.4f}")
        print(f"    Test accuracy: {test_score:.4f}")

        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"    ROC AUC Score: {roc_auc:.4f}")
        except:
            print(f"    ROC AUC Score: N/A (single class in test set)")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"                Predicted")
        print(f"                Legit  Fraud")
        print(f"    Actual Legit  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"           Fraud  {cm[1,0]:5d}  {cm[1,1]:5d}")

        self.models['random_forest'] = rf_model

    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        """Train Gradient Boosting classifier"""
        print(f"\n{'-'*80}")
        print("TRAINING GRADIENT BOOSTING CLASSIFIER")
        print(f"{'-'*80}")

        # Train model
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )

        gb_model.fit(X_train, y_train)

        # Evaluate
        train_score = gb_model.score(X_train, y_train)
        test_score = gb_model.score(X_test, y_test)

        y_pred = gb_model.predict(X_test)
        y_pred_proba = gb_model.predict_proba(X_test)[:, 1]

        print(f"\n  Model Performance:")
        print(f"    Training accuracy: {train_score:.4f}")
        print(f"    Test accuracy: {test_score:.4f}")

        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"    ROC AUC Score: {roc_auc:.4f}")
        except:
            print(f"    ROC AUC Score: N/A")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        self.models['gradient_boosting'] = gb_model

    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression as baseline"""
        print(f"\n{'-'*80}")
        print("TRAINING LOGISTIC REGRESSION (BASELINE)")
        print(f"{'-'*80}")

        # Train model
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        lr_model.fit(X_train, y_train)

        # Evaluate
        train_score = lr_model.score(X_train, y_train)
        test_score = lr_model.score(X_test, y_test)

        print(f"\n  Model Performance:")
        print(f"    Training accuracy: {train_score:.4f}")
        print(f"    Test accuracy: {test_score:.4f}")

        self.models['logistic_regression'] = lr_model

    def create_ensemble_model(self, X_test, y_test):
        """Create ensemble by averaging predictions"""
        print(f"\n{'-'*80}")
        print("CREATING ENSEMBLE MODEL")
        print(f"{'-'*80}")

        # Get predictions from all models
        predictions = []

        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions.append(pred_proba)

        # Average predictions
        ensemble_proba = np.mean(predictions, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        # Evaluate
        from sklearn.metrics import accuracy_score
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

        print(f"\n  Ensemble Performance:")
        print(f"    Test accuracy: {ensemble_accuracy:.4f}")

        try:
            roc_auc = roc_auc_score(y_test, ensemble_proba)
            print(f"    ROC AUC Score: {roc_auc:.4f}")
        except:
            print(f"    ROC AUC Score: N/A")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, ensemble_pred, zero_division=0))

    def analyze_feature_importance(self):
        """Analyze feature importance from Random Forest"""
        print(f"\n{'-'*80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'-'*80}")

        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importances = rf_model.feature_importances_

            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"\n  TOP 15 MOST IMPORTANT FEATURES:")
            print(f"  {'-' * 70}")

            for idx, row in feature_importance.head(15).iterrows():
                print(f"    {row['feature']:30s}: {row['importance']:.4f}")

            # Save feature importance
            feature_importance.to_csv('feature_importance.csv', index=False)
            print(f"\n  ✓ Feature importance saved to: feature_importance.csv")

    def save_models(self):
        """Save trained models to disk"""
        print(f"\n{'-'*80}")
        print("SAVING MODELS")
        print(f"{'-'*80}")

        # Save each model
        for name, model in self.models.items():
            filename = f'model_{name}.pkl'
            joblib.dump(model, filename)
            print(f"  ✓ Saved: {filename}")

        # Save scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        print(f"  ✓ Saved: scaler.pkl")

        # Save label encoders
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print(f"  ✓ Saved: label_encoders.pkl")

        # Save feature columns
        joblib.dump(self.feature_cols, 'feature_columns.pkl')
        print(f"  ✓ Saved: feature_columns.pkl")


def main():
    """Main execution"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    detector = MLFraudDetector(df)
    detector.train_all_models()

    print(f"\n✓ ML model training complete")
    print(f"\nTrained models:")
    print(f"  - Random Forest")
    print(f"  - Gradient Boosting")
    print(f"  - Logistic Regression")
    print(f"  - Ensemble (average)")


if __name__ == "__main__":
    main()
