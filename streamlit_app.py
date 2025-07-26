import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import json
import time

# === CONFIG ===
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"  # <-- 🔒 Replace this with your AbuseIPDB API key
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
TOP_N = 30  # Number of top IPs to check

# === PAGE SETTINGS ===
st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")
st.title("🔐 Threat Intelligence Dashboard + API Enrichment")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # === SEARCH BAR ===
    ip_query = st.text_input("🔎 Search IP/domain/subnet", "")
    filtered_df = df[df['Source IP'].astype(str).str.contains(ip_query, na=False)] if ip_query else df

    # === TRAFFIC BY PROTOCOL ===
    st.subheader("📊 Traffic by Protocol (Total vs Malicious)")
    grouped = filtered_df.groupby(['Protocol', 'Label']).size().reset_index(name='Count')
    fig1 = px.bar(grouped, x='Protocol', y='Count', color='Label', barmode='group')
    st.plotly_chart(fig1, use_container_width=True)

    # === TOP MALICIOUS IPS ===
    st.subheader("🚨 Top Malicious IPs")
    malicious_df = df[df['Label'] != 'BENIGN']
    if ip_query:
        malicious_df = malicious_df[malicious_df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    top_ips = malicious_df['Source IP'].value_counts().nlargest(10).reset_index()
    top_ips.columns = ['Source IP', 'Count']
    fig2 = px.bar(top_ips, x='Source IP', y='Count')
    st.plotly_chart(fig2, use_container_width=True)

    # === DETECTION RATE PIE CHART ===
    st.subheader("📈 Detection Rate (Benign vs Malicious)")
    rate = filtered_df['Label'].value_counts(normalize=True).reset_index()
    rate.columns = ['Label', 'Percentage']
    fig3 = px.pie(rate, values='Percentage', names='Label')
    st.plotly_chart(fig3, use_container_width=True)

    # === INTRUSION TIME SERIES ===
    st.subheader("📆 Intrusion Events Over Time")
    intrusions = df[df['Label'] != 'BENIGN']
    if ip_query:
        intrusions = intrusions[intrusions['Source IP'].astype(str).str.contains(ip_query, na=False)]
    time_df = intrusions.dropna(subset=['Timestamp']).groupby(
        intrusions['Timestamp'].dt.floor('H')
    ).size().reset_index(name='Count')
    fig4 = px.line(time_df, x='Timestamp', y='Count')
    st.plotly_chart(fig4, use_container_width=True)

    # === SEARCH RESULTS TABLE ===
    if ip_query:
        st.subheader(f"🔍 Search Results for '{ip_query}'")
        result_df = filtered_df[['Timestamp', 'Source IP', 'Protocol', 'Label']]
        st.dataframe(result_df.head(10))

    # === ABUSEIPDB API ENRICHMENT ===
    st.subheader("🌐 Enrich Top Malicious IPs with AbuseIPDB")

    if 'Source IP' not in df.columns:
        st.error("❌ 'Source IP' column is missing from the dataset.")
    else:
        if st.button("🔍 Run IP Reputation Check on Top Malicious IPs"):
            # Filter top malicious IPs
            malicious_df = df[df['Label'] != 'BENIGN']
            ip_list = malicious_df['Source IP'].value_counts().head(TOP_N).index.tolist()
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
                        time.sleep(1)  # Respect rate limits
                    except Exception as e:
                        results[ip] = {"error": str(e)}

                # Display results in expandable boxes
                for ip, data in results.items():
                    with st.expander(f"IP: {ip}"):
                        st.json(data)

                # Save results as downloadable JSON
                json_str = json.dumps(results, indent=4)
                st.download_button(
                    label="📥 Download Enrichment Results (JSON)",
                    data=json_str,
                    file_name="malicious_ip_enrichment.json",
                    mime="application/json"
                )
else:
    st.info("👆 Please upload a CSV file to get started.")
