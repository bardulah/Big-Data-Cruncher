"""
REAL-TIME FRAUD DETECTION API
FastAPI service for instant fraud scoring of marketplace listings
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')


# Pydantic models for request/response
class ListingInput(BaseModel):
    """Input model for fraud detection"""
    category: str = Field(..., description="Product category")
    platform: str = Field(..., description="Marketplace platform")
    brand: Optional[str] = Field(None, description="Brand name")
    condition: str = Field(..., description="Item condition")
    current_price: float = Field(..., gt=0, description="Current listing price in EUR")
    original_price: Optional[float] = Field(None, gt=0, description="Original price")
    age_years: float = Field(..., ge=0, description="Item age in years")
    seller_account_age_days: int = Field(..., ge=0, description="Seller account age in days")
    seller_item_count: int = Field(..., ge=1, description="Number of items seller has listed")
    seller_rating: float = Field(..., ge=0, le=5, description="Seller rating 0-5")
    image_count: int = Field(..., ge=0, description="Number of images")
    description_length: int = Field(..., ge=0, description="Description character count")
    days_listed: int = Field(..., ge=0, description="Days since listed")
    view_count: int = Field(..., ge=0, description="Number of views")
    favorite_count: int = Field(..., ge=0, description="Number of favorites")
    price_drops: int = Field(..., ge=0, description="Number of price reductions")
    shipping_offered: bool = Field(..., description="Whether shipping is offered")

    class Config:
        schema_extra = {
            "example": {
                "category": "electronics",
                "platform": "eBay.de",
                "brand": "Apple",
                "condition": "like-new",
                "current_price": 45.99,
                "original_price": 599.00,
                "age_years": 2,
                "seller_account_age_days": 5,
                "seller_item_count": 150,
                "seller_rating": 3.2,
                "image_count": 1,
                "description_length": 25,
                "days_listed": 3,
                "view_count": 850,
                "favorite_count": 2,
                "price_drops": 0,
                "shipping_offered": True
            }
        }


class FraudResponse(BaseModel):
    """Response model for fraud detection"""
    fraud_score: float = Field(..., description="Fraud confidence score 0-100")
    risk_level: str = Field(..., description="EXTREME, HIGH, MEDIUM, or LOW")
    anomaly_score: float = Field(..., description="Statistical anomaly score 0-100")
    explanation: List[str] = Field(..., description="List of fraud indicators found")
    recommendations: List[str] = Field(..., description="Action recommendations")
    processing_time_ms: float = Field(..., description="API processing time")


class FraudDetectionEngine:
    """Core fraud detection logic"""

    def __init__(self):
        self.luxury_brands = ['Gucci', 'Louis Vuitton', 'Prada', 'Supreme',
                             'Apple', 'Herman Miller']

        # Load historical statistics for comparison
        try:
            self.df_historical = pd.read_csv('marketplace_data.csv')
            self.compute_market_baselines()
        except:
            self.df_historical = None

    def compute_market_baselines(self):
        """Compute market average prices by category/condition"""
        self.baselines = {}

        for category in self.df_historical['category'].unique():
            self.baselines[category] = {}
            cat_data = self.df_historical[self.df_historical['category'] == category]

            for condition in self.df_historical['condition'].unique():
                cond_data = cat_data[cat_data['condition'] == condition]
                if len(cond_data) > 0:
                    self.baselines[category][condition] = {
                        'mean_price': cond_data['current_price'].mean(),
                        'median_price': cond_data['current_price'].median(),
                        'std_price': cond_data['current_price'].std()
                    }

    def detect_fraud(self, listing: ListingInput) -> dict:
        """Main fraud detection function"""
        start_time = datetime.now()

        fraud_score = 0.0
        anomaly_score = 0.0
        indicators = []
        recommendations = []

        # Layer 1: Price anomalies
        price_signals = self._check_price_anomalies(listing)
        fraud_score += price_signals['score']
        indicators.extend(price_signals['indicators'])

        # Layer 2: Behavioral anomalies
        behavior_signals = self._check_behavioral_anomalies(listing)
        fraud_score += behavior_signals['score']
        indicators.extend(behavior_signals['indicators'])

        # Layer 3: Content anomalies
        content_signals = self._check_content_anomalies(listing)
        fraud_score += content_signals['score']
        indicators.extend(content_signals['indicators'])

        # Layer 4: Luxury counterfeit detection
        luxury_signals = self._check_luxury_counterfeits(listing)
        fraud_score += luxury_signals['score']
        indicators.extend(luxury_signals['indicators'])

        # Layer 5: Statistical anomalies
        stat_signals = self._check_statistical_anomalies(listing)
        anomaly_score = stat_signals['score']
        indicators.extend(stat_signals['indicators'])

        # Combine scores
        final_fraud_score = min(100, fraud_score)
        final_anomaly_score = min(100, anomaly_score)

        # Risk level
        if final_fraud_score >= 80:
            risk_level = "EXTREME"
            recommendations.append("AUTO-REMOVE: Extremely high fraud probability")
            recommendations.append("Flag seller account for review")
        elif final_fraud_score >= 60:
            risk_level = "HIGH"
            recommendations.append("FLAG FOR MANUAL REVIEW")
            recommendations.append("Request additional verification from seller")
        elif final_fraud_score >= 40:
            risk_level = "MEDIUM"
            recommendations.append("Monitor closely")
            recommendations.append("Check seller history")
        else:
            risk_level = "LOW"
            recommendations.append("Standard processing")

        # Processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            'fraud_score': round(final_fraud_score, 2),
            'risk_level': risk_level,
            'anomaly_score': round(final_anomaly_score, 2),
            'explanation': indicators if indicators else ["No significant fraud indicators detected"],
            'recommendations': recommendations,
            'processing_time_ms': round(processing_time, 2)
        }

    def _check_price_anomalies(self, listing: ListingInput) -> dict:
        """Check for price-based fraud signals"""
        score = 0
        indicators = []

        # Check against market baseline
        if self.df_historical is not None:
            baseline = self.baselines.get(listing.category, {}).get(listing.condition)

            if baseline:
                mean_price = baseline['mean_price']
                std_price = baseline['std_price']

                # Extremely low price
                if listing.current_price < mean_price * 0.1:
                    score += 30
                    indicators.append(
                        f"EXTREME LOW PRICE: €{listing.current_price:.2f} vs "
                        f"market avg €{mean_price:.2f} (90% below market)"
                    )
                elif listing.current_price < mean_price * 0.3:
                    score += 20
                    indicators.append(
                        f"Suspiciously low price: €{listing.current_price:.2f} vs "
                        f"market avg €{mean_price:.2f}"
                    )

                # Extremely high price
                if listing.current_price > mean_price * 5:
                    score += 10
                    indicators.append(f"Unusually high price (5x market average)")

        # Original vs current price check
        if listing.original_price:
            discount = (listing.original_price - listing.current_price) / listing.original_price * 100
            if discount > 90:
                score += 25
                indicators.append(f"Extreme discount: {discount:.0f}% off original price")

        return {'score': score, 'indicators': indicators}

    def _check_behavioral_anomalies(self, listing: ListingInput) -> dict:
        """Check behavioral patterns"""
        score = 0
        indicators = []

        # New seller with bulk listings
        if listing.seller_account_age_days < 30 and listing.seller_item_count > 100:
            score += 25
            indicators.append(
                f"NEW BULK SELLER: {listing.seller_account_age_days}-day account "
                f"with {listing.seller_item_count} listings"
            )

        # High views but no favorites (misleading listing)
        if listing.view_count > 500 and listing.favorite_count < 10:
            score += 15
            indicators.append(
                f"High views ({listing.view_count}) but low favorites ({listing.favorite_count}) "
                f"- possible misleading listing"
            )

        # Low seller rating
        if listing.seller_rating < 2.5 and listing.seller_account_age_days > 90:
            score += 10
            indicators.append(f"Poor seller rating: {listing.seller_rating}/5.0")

        # Rapid price drops
        if listing.price_drops >= 3 and listing.days_listed < 14:
            score += 10
            indicators.append(f"Desperate pricing: {listing.price_drops} drops in {listing.days_listed} days")

        return {'score': score, 'indicators': indicators}

    def _check_content_anomalies(self, listing: ListingInput) -> dict:
        """Check content quality"""
        score = 0
        indicators = []

        # No images
        if listing.image_count == 0:
            score += 15
            indicators.append("NO IMAGES provided")

        # Minimal description
        if listing.description_length < 20:
            score += 10
            indicators.append(f"Minimal description ({listing.description_length} characters)")

        # Combined poor content
        if listing.image_count == 0 and listing.description_length < 50:
            score += 10
            indicators.append("POOR CONTENT: No images AND minimal description")

        return {'score': score, 'indicators': indicators}

    def _check_luxury_counterfeits(self, listing: ListingInput) -> dict:
        """Check for luxury good counterfeits"""
        score = 0
        indicators = []

        if listing.brand in self.luxury_brands:
            # Luxury item from new seller
            if listing.seller_account_age_days < 30:
                score += 20
                indicators.append(
                    f"LUXURY BRAND from new seller ({listing.seller_account_age_days} days old)"
                )

            # Low price luxury
            if self.df_historical is not None:
                luxury_items = self.df_historical[
                    self.df_historical['brand'] == listing.brand
                ]
                if len(luxury_items) > 0:
                    avg_luxury_price = luxury_items['current_price'].mean()

                    if listing.current_price < avg_luxury_price * 0.3:
                        score += 40
                        indicators.append(
                            f"COUNTERFEIT ALERT: {listing.brand} at €{listing.current_price:.2f} "
                            f"(typical: €{avg_luxury_price:.2f}) - 99% fake probability"
                        )

            # Bulk luxury from new seller
            if listing.seller_account_age_days < 30 and listing.seller_item_count > 50:
                score += 25
                indicators.append(
                    f"EXTREME RISK: New seller with bulk {listing.brand} listings - "
                    f"possible counterfeit ring"
                )

        return {'score': score, 'indicators': indicators}

    def _check_statistical_anomalies(self, listing: ListingInput) -> dict:
        """Statistical anomaly detection"""
        score = 0
        indicators = []

        # View-to-favorite ratio
        if listing.view_count > 0:
            fav_rate = listing.favorite_count / listing.view_count
            if fav_rate < 0.01 and listing.view_count > 100:
                score += 20
                indicators.append(f"Abnormal engagement: {fav_rate*100:.1f}% favorite rate")

        # Price per image (for luxury)
        if listing.brand in self.luxury_brands and listing.image_count > 0:
            if listing.image_count < 2:
                score += 15
                indicators.append(f"Luxury item with only {listing.image_count} image(s)")

        # Seller experience vs volume mismatch
        expected_items = (listing.seller_account_age_days / 30) * 2  # ~2 items/month
        if listing.seller_item_count > expected_items * 10:
            score += 10
            indicators.append(
                f"Unusual seller activity: {listing.seller_item_count} items in "
                f"{listing.seller_account_age_days} days"
            )

        return {'score': score, 'indicators': indicators}


# Initialize FastAPI app
app = FastAPI(
    title="Marketplace Fraud Detection API",
    description="Real-time fraud detection for secondhand marketplace listings",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fraud detection engine
fraud_engine = FraudDetectionEngine()


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Marketplace Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect-fraud": "POST - Detect fraud in a listing",
            "/health": "GET - Health check",
            "/stats": "GET - API statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": fraud_engine.df_historical is not None
    }


@app.post("/detect-fraud", response_model=FraudResponse)
async def detect_fraud(listing: ListingInput):
    """
    Detect fraud in a marketplace listing

    Returns fraud score (0-100) and detailed explanation
    """
    try:
        result = fraud_engine.detect_fraud(listing)
        return FraudResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud detection failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    if fraud_engine.df_historical is None:
        return {"message": "No historical data loaded"}

    df = fraud_engine.df_historical

    return {
        "total_items_analyzed": len(df),
        "categories": df['category'].value_counts().to_dict(),
        "platforms": df['platform'].value_counts().to_dict(),
        "avg_fraud_score": df['suspected_counterfeit'].mean() * 100,
        "high_risk_items": (df['suspected_counterfeit'] > 0.75).sum()
    }


@app.post("/batch-detect", response_model=List[FraudResponse])
async def batch_detect_fraud(listings: List[ListingInput]):
    """
    Batch fraud detection for multiple listings

    Maximum 100 listings per request
    """
    if len(listings) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 listings per batch request"
        )

    results = []
    for listing in listings:
        try:
            result = fraud_engine.detect_fraud(listing)
            results.append(FraudResponse(**result))
        except Exception as e:
            # Continue processing other listings even if one fails
            results.append(FraudResponse(
                fraud_score=0,
                risk_level="ERROR",
                anomaly_score=0,
                explanation=[f"Error: {str(e)}"],
                recommendations=["Manual review required"],
                processing_time_ms=0
            ))

    return results


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*80)
    print("STARTING FRAUD DETECTION API")
    print("="*80)
    print("\nAPI Endpoints:")
    print("  - POST http://localhost:8000/detect-fraud")
    print("  - POST http://localhost:8000/batch-detect")
    print("  - GET  http://localhost:8000/health")
    print("  - GET  http://localhost:8000/stats")
    print("\nInteractive API docs:")
    print("  - http://localhost:8000/docs (Swagger UI)")
    print("  - http://localhost:8000/redoc (ReDoc)")
    print("\nStarting server...")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
