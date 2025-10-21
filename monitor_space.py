#!/usr/bin/env python3
"""
🏀 Monitor Hugging Face Space Status
Continuously checks if your space is ready
"""

import requests
import time
from datetime import datetime

SPACE_URL = "https://huggingface.co/spaces/Cossy179/Goon-Steen"

def check_api_ready():
    """Check if the API is ready"""
    try:
        response = requests.get(f"{SPACE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                return True, data
            except:
                return False, "Invalid JSON response"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, str(e)

def monitor_space():
    """Monitor space until it's ready"""
    print("🏀 Monitoring Hugging Face Space...")
    print(f"🌐 Space URL: {SPACE_URL}")
    print("⏰ Checking every 30 seconds...")
    print("🛑 Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{timestamp}] Check #{check_count}: ", end="", flush=True)
            
            is_ready, result = check_api_ready()
            
            if is_ready:
                print("✅ API IS READY!")
                print(f"   Model Status: {result.get('model_status', 'unknown')}")
                print(f"   Cache Valid: {result.get('cache_valid', 'unknown')}")
                print("\n🎉 Your NBA Predictions API is now live!")
                print("💡 You can now run the full test suite:")
                print("   py test_hf_api.py")
                break
            else:
                print(f"⏳ Not ready yet ({result})")
                print("   💡 Check the Hugging Face Space logs for build progress")
            
            print("   ⏰ Waiting 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        print("💡 You can run this script again later to check status")

if __name__ == "__main__":
    monitor_space()







