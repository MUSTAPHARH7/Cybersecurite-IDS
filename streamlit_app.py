import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import json
import time

# === CONFIG ===
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"  # Replace with your AbuseIPDB API key
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
TOP_N = 30  # number of top IPs to check

st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")
st.title("🔐 Threat Intelligence Dashboard + API Enrichment")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # --- Dashboard ---
    st.subheader("📈 Dashboard")
    ip_query = st.text_input("🔎 Search IP/domain/subnet", "")
    filtered_df = df[df['Source IP'].astype(str).str.contains(ip_query, na=False)] if ip_query else df

    grouped = filtered_df.groupby(['Protocol', 'Label']).size().reset_index(name='Count')
    st.plotly_chart(px.bar(grouped, x='Protocol', y='Count', color='Label', barmode='group'), use_container_width=True)

    # --- API Enrichment ---
    st.subheader("🌐 Enrich Top IPs with AbuseIPDB")

    if 'Destination IP' not in df.columns:
        st.error("❌ 'Destination IP' column is missing from the dataset.")
    else:
        if st.button("🔍 Run IP Reputation Check (Top 30)"):
            ip_list = df['Destination IP'].value_counts().head(TOP_N).index.tolist()
            results = {}

            with st.spinner("⏳ Querying AbuseIPDB..."):
                for ip in ip_list:
                    try:
                        response = requests.get(
                            ABUSEIPDB_URL,
                            headers={"Key": API_KEY, "Accept": "application/json"},
                            params={"ipAddress": ip, "maxAgeInDays": 90, "verbose": True}
                        )
                        if response.status_code == 200:
                            results[ip] = response.json()
                        else:
                            results[ip] = {"error": f"Status {response.status_code}", "reason": response.text}
                        time.sleep(1)  # To respect rate limits
                    except Exception as e:
                        results[ip] = {"error": str(e)}

                # Display results in expandable boxes
                for ip, data in results.items():
                    with st.expander(f"IP: {ip}"):
                        st.json(data)

                # Save to JSON in memory
                json_str = json.dumps(results, indent=4)
                st.download_button(
                    label="📥 Download Results as JSON",
                    data=json_str,
                    file_name="api_enriched_threats.json",
                    mime="application/json"
                )

else:
    st.info("👆 Please upload a CSV file to get started.")
