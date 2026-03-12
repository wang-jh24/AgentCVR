"""
Test script for chat completion API connection.
Replace TARGET_IP, PORT, API_KEY, MODEL_NAME with your deployment values.
"""
import requests
import json
import time
import sys

# ==================== Configuration ====================
# Use 127.0.0.1 when running on the same machine as the server
# Use your server IP when accessing remotely
TARGET_IP = "127.0.0.1"
# TARGET_IP = "YOUR_SERVER_IP"

# Ports:
# 8000 -> Load balancer (recommended, distributes across workers)
# 8001 -> Worker 1 (e.g. GPU 0)
# 8002 -> Worker 2 (e.g. GPU 1)
PORT = "8000"

API_URL = f"http://{TARGET_IP}:{PORT}/v1/chat/completions"

# Replace with your deployed model config (e.g. from start_workers.sh)
API_KEY = "your_api_key_here"
MODEL_NAME = "your-model-name"

def test_new_deployment():
    print("Testing API connection...")
    print(f"Target: {API_URL}")
    print(f"Model: {MODEL_NAME}")
    print("API Key: (configured)" if API_KEY and API_KEY != "your_api_key_here" else "API Key: (replace your_api_key_here in script)")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Write a short paragraph about machine learning."
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }

    s = requests.Session()
    # Set to False to ignore HTTP_PROXY/HTTPS_PROXY and avoid proxy intercepting local requests
    s.trust_env = False

    try:
        start_time = time.time()
        print("Sending request...")
        response = s.post(API_URL, headers=headers, json=payload, timeout=10)
        duration = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            choice = result.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "(no content)")

            print("\n" + "=" * 40)
            print(f"OK (took {duration:.2f}s)")
            print("=" * 40)
            print(f"Reply:\n{content.strip()}")
            print("=" * 40)
            print("Tip: Check load_balancer.py console to see which worker handled the request.")
        else:
            print("\n" + "=" * 40)
            print(f"Error (status {response.status_code})")
            print("=" * 40)
            print(f"Details: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"\nConnection failed: cannot reach {API_URL}")
        print("Check:")
        print("1. Is load_balancer.py running? (Try port 8001 for single worker)")
        print("2. Did start_workers.sh run successfully?")
        print("3. Is the IP correct?")
    except Exception as e:
        print(f"\nException: {e}")

if __name__ == "__main__":
    test_new_deployment()
