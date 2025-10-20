#!/usr/bin/env python3
"""
🏀 Test After Update
Run this after uploading the updated files to your Hugging Face Space
"""

import requests
import json
import time

SPACE_URL = "https://huggingface.co/spaces/Cossy179/Goon-Steen"

def test_after_update():
    """Test the space after uploading updated files"""
    print("🏀 Testing Hugging Face Space After Update")
    print(f"🌐 Space URL: {SPACE_URL}")
    print("-" * 60)
    
    # Wait a moment for any final build steps
    print("⏰ Waiting 10 seconds for space to stabilize...")
    time.sleep(10)
    
    # Test the new test endpoint first
    print("\n🔍 Testing new /test endpoint...")
    try:
        response = requests.get(f"{SPACE_URL}/test", timeout=15)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("   ✅ Test endpoint working!")
                print(f"   Message: {data.get('message', 'No message')}")
                print(f"   Model Loaded: {data.get('model_loaded', 'Unknown')}")
                
                if data.get('model_loaded'):
                    print("   🎉 Model is loaded and working!")
                else:
                    print("   ⚠️  Model not loaded, but fallback predictions should work")
                
            except json.JSONDecodeError:
                print("   ❌ Response is not JSON")
                return False
        else:
            print(f"   ❌ Test endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing /test endpoint: {e}")
        return False
    
    # Test root endpoint
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(SPACE_URL, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                try:
                    data = response.json()
                    print("   ✅ Root endpoint returning JSON!")
                    print(f"   Service: {data.get('service', 'Unknown')}")
                    print(f"   Status: {data.get('status', 'Unknown')}")
                except:
                    print("   ⚠️  JSON parse error")
            else:
                print("   ⚠️  Still returning HTML instead of JSON")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error testing root endpoint: {e}")
    
    # Test health endpoint
    print("\n🔍 Testing /api/health endpoint...")
    try:
        response = requests.get(f"{SPACE_URL}/api/health", timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("   ✅ Health endpoint working!")
                print(f"   Status: {data.get('status', 'Unknown')}")
                print(f"   Model Status: {data.get('model_status', 'Unknown')}")
                print(f"   Cache Valid: {data.get('cache_valid', 'Unknown')}")
            except:
                print("   ⚠️  JSON parse error")
        else:
            print(f"   ❌ Health endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error testing health endpoint: {e}")
    
    # Test predictions endpoint
    print("\n🔍 Testing /api/predictions endpoint...")
    try:
        response = requests.get(f"{SPACE_URL}/api/predictions", timeout=15)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("   ✅ Predictions endpoint working!")
                print(f"   Date: {data.get('date', 'Unknown')}")
                print(f"   Total Games: {data.get('total_games', 0)}")
                print(f"   Total Parlays: {data.get('total_parlays', 0)}")
                
                if data.get('total_games', 0) > 0:
                    print("   🎮 Games found for today!")
                else:
                    print("   ℹ️  No games for today (normal if no NBA games scheduled)")
                    
            except:
                print("   ⚠️  JSON parse error")
        else:
            print(f"   ❌ Predictions endpoint failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Error testing predictions endpoint: {e}")
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print("✅ If you see successful responses above, your API is working!")
    print("💡 You can now run the full test suite: py test_hf_api.py")
    print("🔗 Your API is ready for integration with your web application")

if __name__ == "__main__":
    test_after_update()






