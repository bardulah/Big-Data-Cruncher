"""
PHASE 9: NLP DESCRIPTION ANALYSIS
Analyzes listing descriptions for fraud patterns, sentiment, and keyword analysis
(Simulated NLP - in production would use NLTK/spaCy/transformers)
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class NLPAnalyzer:
    """Analyzes text descriptions for fraud patterns"""

    def __init__(self, df):
        self.df = df.copy()

        # Fraud-related keywords (would use ML models in production)
        self.urgency_keywords = [
            'urgent', 'must sell', 'final price', 'last chance', 'today only',
            'limited time', 'act now', 'don\'t miss', 'hurry', 'quick sale'
        ]

        self.authenticity_keywords = [
            'not authentic', 'replica', 'may be fake', 'copy', 'imitation',
            'not original', 'might be counterfeit', 'unsure if real'
        ]

        self.quality_keywords = [
            'excellent', 'perfect', 'flawless', 'mint', 'brand new',
            'never used', 'pristine', 'immaculate'
        ]

        self.negative_keywords = [
            'damaged', 'broken', 'scratched', 'worn', 'stained',
            'defective', 'issue', 'problem', 'flaw'
        ]

    def analyze_all_descriptions(self):
        """Perform complete NLP analysis"""
        print("\n" + "="*80)
        print("PHASE 9: NLP DESCRIPTION ANALYSIS")
        print("="*80)

        # Simulate descriptions based on existing data
        self.generate_synthetic_descriptions()

        # Keyword analysis
        self.keyword_fraud_detection()

        # Sentiment analysis (simulated)
        self.sentiment_analysis()

        # Description quality assessment
        self.description_quality_analysis()

        # Language pattern detection
        self.detect_language_patterns()

        print("\n" + "="*80)
        print("PHASE 9 COMPLETE")
        print("="*80)

    def generate_synthetic_descriptions(self):
        """Generate synthetic descriptions based on item attributes"""
        print(f"\n{'-'*80}")
        print("GENERATING SYNTHETIC DESCRIPTIONS FOR ANALYSIS")
        print(f"{'-'*80}")

        descriptions = []

        for idx, row in self.df.iterrows():
            desc_parts = []

            # Add category and brand
            if row['brand'] and pd.notna(row['brand']):
                desc_parts.append(f"{row['brand']} {row['category']}")
            else:
                desc_parts.append(f"{row['category']}")

            # Add condition
            desc_parts.append(f"in {row['condition']} condition")

            # Add urgency for high fraud score items
            if row.get('suspected_counterfeit', 0) > 0.7:
                desc_parts.append("MUST SELL TODAY! URGENT!")

            # Add quality claims (sometimes exaggerated for fraud)
            if row['condition'] == 'mint':
                desc_parts.append("Perfect condition, never used")

            # Add price mention
            if row.get('price_drops', 0) > 2:
                desc_parts.append("FINAL PRICE REDUCTION!")

            # Create description
            description = ". ".join(desc_parts) + "."

            # Adjust length
            if row['description_length'] < 50:
                description = description[:row['description_length']]

            descriptions.append(description)

        self.df['description_text'] = descriptions

        print(f"  ✓ Generated {len(descriptions):,} synthetic descriptions")
        print(f"\n  Sample descriptions:")
        for i in range(min(5, len(self.df))):
            print(f"    Item {i}: {self.df.iloc[i]['description_text'][:80]}...")

    def keyword_fraud_detection(self):
        """Detect fraud-related keywords in descriptions"""
        print(f"\n{'-'*80}")
        print("KEYWORD-BASED FRAUD DETECTION")
        print(f"{'-'*80}")

        # Count keywords
        self.df['urgency_score'] = self.df['description_text'].apply(
            lambda x: sum(1 for word in self.urgency_keywords if word.lower() in str(x).lower())
        )

        self.df['authenticity_concern_score'] = self.df['description_text'].apply(
            lambda x: sum(1 for word in self.authenticity_keywords if word.lower() in str(x).lower())
        )

        self.df['quality_claims_score'] = self.df['description_text'].apply(
            lambda x: sum(1 for word in self.quality_keywords if word.lower() in str(x).lower())
        )

        # Analysis
        urgency_items = self.df[self.df['urgency_score'] > 0]
        auth_concern_items = self.df[self.df['authenticity_concern_score'] > 0]

        print(f"\n  KEYWORD DETECTION RESULTS:")
        print(f"    Items with urgency keywords: {len(urgency_items)} ({len(urgency_items)/len(self.df)*100:.1f}%)")
        print(f"    Items with authenticity concerns: {len(auth_concern_items)} ({len(auth_concern_items)/len(self.df)*100:.1f}%)")

        # Correlation with fraud
        if len(urgency_items) > 0:
            urgency_fraud_rate = urgency_items['suspected_counterfeit'].mean()
            normal_fraud_rate = self.df[self.df['urgency_score'] == 0]['suspected_counterfeit'].mean()

            print(f"\n  FRAUD CORRELATION:")
            print(f"    Avg fraud score (urgency keywords): {urgency_fraud_rate:.3f}")
            print(f"    Avg fraud score (no urgency): {normal_fraud_rate:.3f}")
            print(f"    → Urgency keywords increase fraud probability by {(urgency_fraud_rate/normal_fraud_rate - 1)*100:.1f}%")

        # Items with authenticity concerns
        if len(auth_concern_items) > 0:
            print(f"\n  AUTHENTICITY CONCERNS (honest admissions):")
            print(f"    Items mentioning potential fakes: {len(auth_concern_items)}")
            print(f"    These sellers are being honest but still risky!")

    def sentiment_analysis(self):
        """Analyze sentiment of descriptions (simulated)"""
        print(f"\n{'-'*80}")
        print("SENTIMENT ANALYSIS")
        print(f"{'-'*80}")

        # Simulate sentiment scoring
        # In production: use VADER, TextBlob, or transformers

        def calculate_sentiment(text):
            """Simulated sentiment score -1 to 1"""
            text_lower = str(text).lower()

            # Positive indicators
            positive_count = sum(1 for word in self.quality_keywords if word in text_lower)

            # Negative indicators
            negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

            # Urgency (slightly negative)
            urgency_count = sum(1 for word in self.urgency_keywords if word in text_lower)

            # Calculate score
            score = (positive_count - negative_count - urgency_count * 0.5)
            return np.clip(score / 5, -1, 1)  # Normalize to -1 to 1

        self.df['sentiment_score'] = self.df['description_text'].apply(calculate_sentiment)

        # Analysis
        print(f"\n  SENTIMENT DISTRIBUTION:")
        print(f"    Mean sentiment: {self.df['sentiment_score'].mean():.3f}")
        print(f"    Median sentiment: {self.df['sentiment_score'].median():.3f}")

        # Sentiment by sold status
        sold_sentiment = self.df[self.df['sold'] == 'yes']['sentiment_score'].mean()
        unsold_sentiment = self.df[self.df['sold'] == 'no']['sentiment_score'].mean()

        print(f"\n  SENTIMENT vs SALE SUCCESS:")
        print(f"    Sold items sentiment: {sold_sentiment:.3f}")
        print(f"    Unsold items sentiment: {unsold_sentiment:.3f}")

        if sold_sentiment > unsold_sentiment:
            print(f"    → Positive descriptions sell {((sold_sentiment - unsold_sentiment) / unsold_sentiment * 100):.1f}% better")

        # Overly positive (might be fake)
        overly_positive = self.df[self.df['sentiment_score'] > 0.8]
        if len(overly_positive) > 0:
            print(f"\n  OVERLY POSITIVE DESCRIPTIONS (potential overselling):")
            print(f"    Count: {len(overly_positive)}")
            print(f"    Avg fraud score: {overly_positive['suspected_counterfeit'].mean():.3f}")

    def description_quality_analysis(self):
        """Analyze description quality metrics"""
        print(f"\n{'-'*80}")
        print("DESCRIPTION QUALITY ASSESSMENT")
        print(f"{'-'*80}")

        # Quality metrics
        self.df['word_count'] = self.df['description_text'].apply(
            lambda x: len(str(x).split())
        )

        self.df['avg_word_length'] = self.df['description_text'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )

        # All caps (shouting)
        self.df['all_caps_ratio'] = self.df['description_text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )

        # Quality assessment
        print(f"\n  DESCRIPTION QUALITY METRICS:")
        print(f"    Avg word count: {self.df['word_count'].mean():.1f}")
        print(f"    Avg word length: {self.df['avg_word_length'].mean():.2f} chars")
        print(f"    Avg caps ratio: {self.df['all_caps_ratio'].mean():.3f}")

        # All caps analysis
        excessive_caps = self.df[self.df['all_caps_ratio'] > 0.3]
        print(f"\n  EXCESSIVE CAPS (potential spam):")
        print(f"    Items with >30% caps: {len(excessive_caps)}")
        if len(excessive_caps) > 0:
            print(f"    Avg fraud score: {excessive_caps['suspected_counterfeit'].mean():.3f}")

        # Minimal descriptions
        minimal = self.df[self.df['word_count'] < 5]
        print(f"\n  MINIMAL DESCRIPTIONS (<5 words):")
        print(f"    Count: {len(minimal)}")
        print(f"    Sale rate: {(minimal['sold'] == 'yes').mean() * 100:.1f}%")

    def detect_language_patterns(self):
        """Detect suspicious language patterns"""
        print(f"\n{'-'*80}")
        print("LANGUAGE PATTERN DETECTION")
        print(f"{'-'*80}")

        # Excessive punctuation
        self.df['exclamation_count'] = self.df['description_text'].apply(
            lambda x: str(x).count('!')
        )

        excessive_excl = self.df[self.df['exclamation_count'] > 3]

        print(f"\n  EXCESSIVE PUNCTUATION:")
        print(f"    Items with >3 exclamation marks: {len(excessive_excl)}")
        if len(excessive_excl) > 0:
            print(f"    Avg fraud score: {excessive_excl['suspected_counterfeit'].mean():.3f}")
            print(f"    → Aggressive language often correlates with scams")

        # Repeated words (spam pattern)
        def has_repeated_words(text):
            words = str(text).lower().split()
            return len(words) != len(set(words))

        self.df['has_repetition'] = self.df['description_text'].apply(has_repeated_words)
        repetitive = self.df[self.df['has_repetition']]

        print(f"\n  WORD REPETITION:")
        print(f"    Items with repeated words: {len(repetitive)}")

        # Save NLP features
        nlp_features = self.df[[
            'item_id', 'urgency_score', 'authenticity_concern_score',
            'sentiment_score', 'word_count', 'all_caps_ratio',
            'exclamation_count'
        ]]

        nlp_features.to_csv('nlp_features.csv', index=False)
        print(f"\n  ✓ NLP features saved to: nlp_features.csv")


def main():
    """Main execution"""
    print("\nLoading marketplace data...")
    df = pd.read_csv('marketplace_data.csv')
    print(f"✓ Loaded {len(df):,} records")

    analyzer = NLPAnalyzer(df)
    analyzer.analyze_all_descriptions()

    print(f"\n✓ NLP analysis complete")


if __name__ == "__main__":
    main()
