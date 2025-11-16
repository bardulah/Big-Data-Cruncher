"""
API TEST CLIENT
Test the fraud detection API with sample requests
"""

import requests
import json
from pprint import pprint


API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("TESTING HEALTH ENDPOINT")
    print("="*80)

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


def test_stats():
    """Test stats endpoint"""
    print("\n" + "="*80)
    print("TESTING STATS ENDPOINT")
    print("="*80)

    response = requests.get(f"{API_BASE_URL}/stats")
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


def test_fraud_detection_legitimate():
    """Test with a legitimate listing"""
    print("\n" + "="*80)
    print("TEST 1: LEGITIMATE LISTING")
    print("="*80)

    listing = {
        "category": "electronics",
        "platform": "eBay.de",
        "brand": "Samsung",
        "condition": "good",
        "current_price": 299.99,
        "original_price": 599.00,
        "age_years": 2,
        "seller_account_age_days": 730,  # 2 years
        "seller_item_count": 45,
        "seller_rating": 4.7,
        "image_count": 5,
        "description_length": 350,
        "days_listed": 14,
        "view_count": 125,
        "favorite_count": 18,
        "price_drops": 1,
        "shipping_offered": True
    }

    print(f"\nSending request...")
    response = requests.post(
        f"{API_BASE_URL}/detect-fraud",
        json=listing
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResponse:")
    result = response.json()
    pprint(result)

    print(f"\n📊 VERDICT: {result['risk_level']}")
    print(f"   Fraud Score: {result['fraud_score']}/100")
    print(f"   Processing Time: {result['processing_time_ms']}ms")


def test_fraud_detection_suspicious():
    """Test with a suspicious listing (counterfeit)"""
    print("\n" + "="*80)
    print("TEST 2: SUSPICIOUS LISTING (Potential Counterfeit)")
    print("="*80)

    listing = {
        "category": "electronics",
        "platform": "Facebook Marketplace",
        "brand": "Apple",  # Luxury brand
        "condition": "mint",
        "current_price": 45.99,  # Way too cheap!
        "original_price": 599.00,
        "age_years": 0,
        "seller_account_age_days": 5,  # New seller
        "seller_item_count": 150,  # Bulk listings
        "seller_rating": 3.2,
        "image_count": 1,  # Only 1 image
        "description_length": 25,  # Short description
        "days_listed": 3,
        "view_count": 850,  # High views
        "favorite_count": 2,  # But low favorites
        "price_drops": 0,
        "shipping_offered": True
    }

    print(f"\nSending request...")
    response = requests.post(
        f"{API_BASE_URL}/detect-fraud",
        json=listing
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResponse:")
    result = response.json()
    pprint(result)

    print(f"\n📊 VERDICT: {result['risk_level']}")
    print(f"   Fraud Score: {result['fraud_score']}/100")
    print(f"   Processing Time: {result['processing_time_ms']}ms")
    print(f"\n⚠️  INDICATORS:")
    for indicator in result['explanation']:
        print(f"   - {indicator}")


def test_fraud_detection_extreme():
    """Test with an extreme fraud case"""
    print("\n" + "="*80)
    print("TEST 3: EXTREME FRAUD CASE")
    print("="*80)

    listing = {
        "category": "clothing",
        "platform": "OLX",
        "brand": "Gucci",  # Luxury brand
        "condition": "mint",
        "current_price": 15.00,  # Ridiculously cheap
        "original_price": 1200.00,
        "age_years": 0,
        "seller_account_age_days": 2,  # Brand new seller
        "seller_item_count": 250,  # Massive bulk
        "seller_rating": 2.1,  # Low rating
        "image_count": 0,  # NO IMAGES!
        "description_length": 12,  # Minimal description
        "days_listed": 1,
        "view_count": 1250,
        "favorite_count": 0,
        "price_drops": 0,
        "shipping_offered": False
    }

    print(f"\nSending request...")
    response = requests.post(
        f"{API_BASE_URL}/detect-fraud",
        json=listing
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResponse:")
    result = response.json()
    pprint(result)

    print(f"\n📊 VERDICT: {result['risk_level']}")
    print(f"   Fraud Score: {result['fraud_score']}/100")
    print(f"   Processing Time: {result['processing_time_ms']}ms")
    print(f"\n🚨 RED FLAGS:")
    for indicator in result['explanation']:
        print(f"   - {indicator}")
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(f"   - {rec}")


def test_batch_detection():
    """Test batch fraud detection"""
    print("\n" + "="*80)
    print("TEST 4: BATCH DETECTION (3 listings)")
    print("="*80)

    listings = [
        {  # Legitimate
            "category": "books",
            "platform": "eBay.de",
            "brand": None,
            "condition": "good",
            "current_price": 12.50,
            "original_price": 19.99,
            "age_years": 3,
            "seller_account_age_days": 450,
            "seller_item_count": 22,
            "seller_rating": 4.5,
            "image_count": 3,
            "description_length": 180,
            "days_listed": 21,
            "view_count": 45,
            "favorite_count": 7,
            "price_drops": 1,
            "shipping_offered": True
        },
        {  # Medium risk
            "category": "electronics",
            "platform": "Vinted",
            "brand": "Apple",
            "condition": "like-new",
            "current_price": 250.00,
            "original_price": 499.00,
            "age_years": 1,
            "seller_account_age_days": 25,
            "seller_item_count": 85,
            "seller_rating": 3.8,
            "image_count": 2,
            "description_length": 95,
            "days_listed": 7,
            "view_count": 320,
            "favorite_count": 12,
            "price_drops": 2,
            "shipping_offered": True
        },
        {  # High risk
            "category": "clothing",
            "platform": "Facebook Marketplace",
            "brand": "Louis Vuitton",
            "condition": "mint",
            "current_price": 35.00,
            "original_price": 1500.00,
            "age_years": 0,
            "seller_account_age_days": 4,
            "seller_item_count": 180,
            "seller_rating": 2.5,
            "image_count": 1,
            "description_length": 18,
            "days_listed": 2,
            "view_count": 950,
            "favorite_count": 1,
            "price_drops": 0,
            "shipping_offered": True
        }
    ]

    print(f"\nSending batch request with {len(listings)} listings...")
    response = requests.post(
        f"{API_BASE_URL}/batch-detect",
        json=listings
    )

    print(f"\nStatus Code: {response.status_code}")
    results = response.json()

    print(f"\n📊 BATCH RESULTS:")
    for i, result in enumerate(results, 1):
        print(f"\n  Listing {i}:")
        print(f"    Risk Level: {result['risk_level']}")
        print(f"    Fraud Score: {result['fraud_score']}/100")
        print(f"    Processing Time: {result['processing_time_ms']}ms")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" "*20 + "FRAUD DETECTION API - TEST SUITE")
    print("="*80)
    print("\nMake sure the API is running on http://localhost:8000")
    print("Start server with: python api_fraud_detection.py")
    print("\nPress Enter to continue...")
    input()

    try:
        # Run tests
        test_health()
        test_stats()
        test_fraud_detection_legitimate()
        test_fraud_detection_suspicious()
        test_fraud_detection_extreme()
        test_batch_detection()

        print("\n" + "="*80)
        print(" "*25 + "ALL TESTS COMPLETED!")
        print("="*80)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("   Make sure the server is running on http://localhost:8000")
        print("   Start with: python api_fraud_detection.py")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
