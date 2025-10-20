#!/usr/bin/env python3
"""
🏀 Hugging Face Space Diagnostic Tool
Helps diagnose issues with your NBA Predictions API deployment
"""

import requests
import time
import json

SPACE_URL = "https://huggingface.co/spaces/Cossy179/Goon-Steen"

def check_space_status():
    """Check the current status of your Hugging Face Space"""
    print("🔍 Diagnosing Hugging Face Space...")
    print(f"🌐 Space URL: {SPACE_URL}")
    
    # Test different endpoints
    endpoints_to_test = [
        ("/", "Root endpoint"),
        ("/api/health", "Health check"),
        ("/api/predictions", "Predictions endpoint"),
        ("/docs", "FastAPI docs"),
        ("/redoc", "FastAPI redoc")
    ]
    
    results = {}
    
    for endpoint, description in endpoints_to_test:
        url = f"{SPACE_URL}{endpoint}"
        print(f"\n🔍 Testing {description}: {endpoint}")
        
        try:
            response = requests.get(url, timeout=10)
            results[endpoint] = {
                'status': response.status_code,
                'content_type': response.headers.get('content-type', 'unknown'),
                'is_json': False,
                'is_html': False,
                'content_preview': ''
            }
            
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'application/json' in content_type:
                results[endpoint]['is_json'] = True
                try:
                    data = response.json()
                    results[endpoint]['content_preview'] = json.dumps(data, indent=2)[:200]
                    print(f"   ✅ JSON Response")
                except:
                    print(f"   ⚠️  JSON parse error")
            elif 'text/html' in content_type:
                results[endpoint]['is_html'] = True
                results[endpoint]['content_preview'] = response.text[:200]
                print(f"   📄 HTML Response")
            else:
                results[endpoint]['content_preview'] = response.text[:200]
                print(f"   📝 Text Response")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout")
            results[endpoint] = {'status': 'timeout', 'error': 'Request timed out'}
        except requests.exceptions.ConnectionError:
            print(f"   🔌 Connection Error")
            results[endpoint] = {'status': 'connection_error', 'error': 'Connection failed'}
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[endpoint] = {'status': 'error', 'error': str(e)}
    
    return results

def analyze_results(results):
    """Analyze the test results and provide recommendations"""
    print("\n" + "="*60)
    print("📊 DIAGNOSIS RESULTS")
    print("="*60)
    
    # Check if FastAPI is running
    api_endpoints = ['/api/health', '/api/predictions', '/docs', '/redoc']
    api_working = any(
        results.get(endpoint, {}).get('status') == 200 and 
        results.get(endpoint, {}).get('is_json', False)
        for endpoint in api_endpoints
    )
    
    if api_working:
        print("✅ FastAPI is running correctly!")
        working_endpoints = [
            endpoint for endpoint in api_endpoints 
            if results.get(endpoint, {}).get('status') == 200
        ]
        print(f"   Working endpoints: {', '.join(working_endpoints)}")
    else:
        print("❌ FastAPI is not running or not accessible")
        
        # Check if it's still building
        root_status = results.get('/', {}).get('status')
        if root_status == 200 and results.get('/', {}).get('is_html'):
            print("   🔄 Space is accessible but showing HTML (likely still building)")
            print("   💡 Wait 2-3 minutes for Docker container to start")
        elif root_status == 200:
            print("   🔄 Space is accessible but API not responding")
            print("   💡 Check Hugging Face Space logs for build errors")
        else:
            print("   ❌ Space is not accessible")
            print("   💡 Check your space URL and make sure it's public")
    
    # Specific recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not api_working:
        print("1. 🔍 Check Hugging Face Space Logs:")
        print("   - Go to your space page")
        print("   - Click the 'Logs' tab")
        print("   - Look for build errors or startup issues")
        
        print("\n2. ⏰ Wait for Build to Complete:")
        print("   - Docker builds can take 2-5 minutes")
        print("   - Look for 'Space is ready' message in logs")
        
        print("\n3. 🔧 Common Issues:")
        print("   - Missing dependencies in requirements.txt")
        print("   - Model files not found")
        print("   - Port configuration issues")
        print("   - Memory limits exceeded")
    
    # Show detailed results
    print("\n📋 DETAILED RESULTS:")
    for endpoint, result in results.items():
        if 'error' in result:
            print(f"   {endpoint}: {result['error']}")
        else:
            status = result.get('status', 'unknown')
            content_type = result.get('content_type', 'unknown')
            print(f"   {endpoint}: {status} ({content_type})")

def main():
    """Main diagnostic function"""
    print("🏀 NBA Predictions API - Space Diagnostic Tool")
    print("="*60)
    
    results = check_space_status()
    analyze_results(results)
    
    print("\n🔄 You can run this script again to check if the space has started")
    print("💡 If issues persist, check the Hugging Face Space logs for detailed error messages")

if __name__ == "__main__":
    main()






