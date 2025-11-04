import pandas as pd
import requests
import json
import time

# === CONFIG ===
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
DATA_PATH = "../dashboard/dashboard_data.csv"
OUTPUT_PATH = "api_enriched_threats.json"
TOP_N = 30  # number of top IPs to check

# === LOAD IPs ===
df = pd.read_csv(DATA_PATH)

if 'Destination IP' not in df.columns:
    raise ValueError("Column 'Destination IP' not found in the dataset.")

ip_list = df['Destination IP'].value_counts().head(TOP_N).index.tolist()

# === ENRICH IPs ===
results = {}

for ip in ip_list:
    print(f"🔍 Checking IP: {ip}")
    try:
        response = requests.get(
            ABUSEIPDB_URL,
            headers={
                "Key": API_KEY,
                "Accept": "application/json"
            },
            params={
                "ipAddress": ip,
                "maxAgeInDays": 90,
                "verbose": True
            }
        )
        if response.status_code == 200:
            results[ip] = response.json()
        else:
            results[ip] = {"error": f"Status code: {response.status_code}", "reason": response.text}

        time.sleep(1)  # respect rate limits
    except Exception as e:
        results[ip] = {"error": str(e)}

# === SAVE OUTPUT ===
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Done. Results saved to {OUTPUT_PATH}")
