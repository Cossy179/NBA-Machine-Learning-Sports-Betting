"""
Quick test script to verify HuggingFace Space locally before deployment
"""
import requests
import json

print("🏀 Testing NBA Predictions API locally...")
print("=" * 60)

BASE_URL = "http://localhost:7860"

def test_endpoint(endpoint, method="GET"):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n Testing {method} {endpoint}")
    print("-" * 60)
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"Response preview: {json.dumps(data, indent=2)[:500]}...")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Is the server running?")
        print("   Start with: uvicorn app:app --reload --port 7860")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n Make sure to start the server first:")
    print("   cd hf_space")
    print("   uvicorn app:app --reload --port 7860\n")
    
    # Run tests
    results = []
    
    results.append(("Health Check", test_endpoint("/")))
    results.append(("API Health", test_endpoint("/api/health")))
    results.append(("Cache Status", test_endpoint("/api/cache-status")))
    results.append(("Predictions", test_endpoint("/api/predictions")))
    results.append(("Games Only", test_endpoint("/api/games")))
    results.append(("Parlays Only", test_endpoint("/api/parlays")))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to deploy to HuggingFace!")
    else:
        print("\n⚠️ Some tests failed. Fix issues before deploying.")

