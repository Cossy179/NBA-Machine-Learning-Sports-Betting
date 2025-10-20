#!/usr/bin/env python3
"""
🏀 Test Current Hugging Face Space State
Tests the current state of your space to see what's working
"""

import requests
import json

SPACE_URL = "https://huggingface.co/spaces/Cossy179/Goon-Steen"

def test_current_state():
    """Test the current state of the space"""
    print("🏀 Testing Current Hugging Face Space State")
    print(f"🌐 Space URL: {SPACE_URL}")
    print("-" * 60)
    
    # Test endpoints
    endpoints = [
        ("/", "Root endpoint"),
        ("/test", "Test endpoint"),
        ("/api/health", "Health check"),
        ("/api/predictions", "Predictions"),
        ("/docs", "FastAPI docs")
    ]
    
    results = {}
    
    for endpoint, description in endpoints:
        url = f"{SPACE_URL}{endpoint}"
        print(f"\n🔍 Testing {description}: {endpoint}")
        
        try:
            response = requests.get(url, timeout=10)
            status = response.status_code
            content_type = response.headers.get('content-type', 'unknown')
            
            print(f"   Status: {status}")
            print(f"   Content-Type: {content_type}")
            
            if status == 200:
                if 'application/json' in content_type:
                    try:
                        data = response.json()
                        print(f"   ✅ JSON Response: {json.dumps(data, indent=2)[:200]}...")
                        results[endpoint] = {'status': 'success', 'data': data}
                    except:
                        print(f"   ⚠️  JSON parse error")
                        results[endpoint] = {'status': 'json_error'}
                else:
                    print(f"   📄 Non-JSON response (likely HTML)")
                    results[endpoint] = {'status': 'html_response'}
            else:
                print(f"   ❌ HTTP {status}")
                results[endpoint] = {'status': 'error', 'code': status}
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout")
            results[endpoint] = {'status': 'timeout'}
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[endpoint] = {'status': 'error', 'message': str(e)}
    
    # Analysis
    print("\n" + "="*60)
    print("📊 ANALYSIS")
    print("="*60)
    
    # Check if FastAPI is working
    fastapi_working = any(
        results.get(endpoint, {}).get('status') == 'success'
        for endpoint in ['/', '/test', '/api/health', '/docs']
    )
    
    if fastapi_working:
        print("✅ FastAPI is working!")
        
        # Check which endpoints are working
        working_endpoints = [
            endpoint for endpoint, result in results.items()
            if result.get('status') == 'success'
        ]
        print(f"   Working endpoints: {', '.join(working_endpoints)}")
        
        # Check model status
        if '/test' in results and results['/test'].get('status') == 'success':
            model_loaded = results['/test']['data'].get('model_loaded', False)
            if model_loaded:
                print("✅ Model is loaded and working!")
            else:
                print("⚠️  Model is not loaded (using fallback predictions)")
        
    else:
        print("❌ FastAPI is not working")
        print("   💡 The space may still be building or there's a configuration issue")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not fastapi_working:
        print("1. 🔄 Wait for the space to finish building (check logs)")
        print("2. 🔧 Re-upload your files if there are build errors")
        print("3. 📋 Check the Hugging Face Space logs for detailed errors")
    else:
        print("1. ✅ Your API is working! You can now use it")
        print("2. 🧪 Run the full test suite: py test_hf_api.py")
        print("3. 🔗 Integrate with your web application")
    
    return results

if __name__ == "__main__":
    test_current_state()






