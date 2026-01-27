import requests
import os
import urllib3

# Disable annoying SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# PASTE YOUR KEY HERE
API_KEY = "cfc42745859368e3d9c8252b457b09fb" 

def brute_force_connect():
    print("🔥 Attempting to bypass Network Block...")
    
    # 1. AGGRESSIVE: Delete System Proxy Variables
    # If your laptop thinks it needs a proxy, this forces it to stop asking.
    for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']:
        if var in os.environ:
            print(f"   ⚠️ Deleting Ghost Proxy: {var}")
            del os.environ[var]
    
    # 2. Masquerade as a Browser (Anti-Firewall)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query=Avengers"
    
    try:
        print("   🚀 Sending Request (Direct Connection)...")
        # verify=False ignores SSL cert errors (common in restricted networks)
        response = requests.get(url, headers=headers, timeout=10, proxies={}, verify=False)
        
        print(f"   📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("\n✅ SUCCESS! We broke through.")
            print("   Action: You can now run 'fetch_features_safe.py'.")
            print("   (I have added this 'Proxy Clear' logic to the safe script for you below).")
        else:
            print(f"❌ Failed with Code: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\n❌ STILL BLOCKED: {e}")
        print("\n👇 FINAL OPTION: DNS CHANGE 👇")
        print("Your Jio SIM is blocking this specific site via DNS.")
        print("1. Press Windows Key.")
        print("2. Type 'View Network Connections'.")
        print("3. Right-click your Wi-Fi > Properties > IPv4 > Properties.")
        print("4. Set DNS to: 8.8.8.8 and 8.8.4.4 (Google DNS).")
        print("5. Disconnect/Reconnect Wi-Fi and try again.")

if __name__ == "__main__":
    brute_force_connect()