#!/usr/bin/env python3
"""
🏀 Quick NBA API Test
Simple test to check if your Hugging Face Space is running
"""

import requests
import json

# Update this URL with your actual Hugging Face Space URL
SPACE_URL = "https://huggingface.co/spaces/Cossy179/Goon-Steen"

def quick_test():
    """Quick test of the API"""
    print("🏀 Testing NBA Predictions API...")
    print(f"🌐 Space URL: {SPACE_URL}")
    
    try:
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(SPACE_URL, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Root endpoint is accessible")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
            return False
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        health_url = f"{SPACE_URL}/api/health"
        response = requests.get(health_url, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   Model Status: {data.get('model_status', 'unknown')}")
            print(f"   Cache Valid: {data.get('cache_valid', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test predictions endpoint
        print("\n3. Testing predictions endpoint...")
        predictions_url = f"{SPACE_URL}/api/predictions"
        response = requests.get(predictions_url, timeout=15)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Predictions endpoint working")
            print(f"   Games: {data.get('total_games', 0)}")
            print(f"   Parlays: {data.get('total_parlays', 0)}")
        else:
            print(f"   ❌ Predictions failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
        
        print("\n🎉 All tests passed! Your API is working correctly.")
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out - the space might still be starting up")
        print("💡 Hugging Face Spaces can take 1-2 minutes to start")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - check your space URL")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your Hugging Face Space is running")
        print("2. Check the space URL is correct")
        print("3. Wait 1-2 minutes for the space to fully start")
        print("4. Check the Hugging Face Space logs for errors")







