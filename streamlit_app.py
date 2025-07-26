import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import json
import time
from fpdf import FPDF
from io import BytesIO

# === CONFIG ===
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"  # Replace with your valid API key
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
TOP_N = 30

st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")
st.title("🔐 Threat Intelligence Dashboard + API Enrichment")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    ip_query = st.text_input("🔎 Search IP/domain/subnet", "")

    if ip_query:
        filtered_df = df[df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    else:
        filtered_df = df

    # --- Traffic by Protocol ---
    st.subheader("📊 Traffic by Protocol (Total vs Malicious)")
    grouped = filtered_df.groupby(['Protocol', 'Label']).size().reset_index(name='Count')
    fig1 = px.bar(grouped, x='Protocol', y='Count', color='Label', barmode='group')
    st.plotly_chart(fig1, use_container_width=True)

    # --- Top Malicious IPs ---
    st.subheader("🚨 Top Malicious IPs")
    malicious_df = df[df['Label'] != 'BENIGN']
    if ip_query:
        malicious_df = malicious_df[malicious_df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    top_ips = malicious_df['Source IP'].value_counts().nlargest(10).reset_index()
    top_ips.columns = ['Source IP', 'Count']
    fig2 = px.bar(top_ips, x='Source IP', y='Count')
    st.plotly_chart(fig2, use_container_width=True)

    # --- Detection Rate Pie ---
    st.subheader("📈 Detection Rate (Benign vs Malicious)")
    rate = filtered_df['Label'].value_counts(normalize=True).reset_index()
    rate.columns = ['Label', 'Percentage']
    fig3 = px.pie(rate, values='Percentage', names='Label')
    st.plotly_chart(fig3, use_container_width=True)

    # --- Intrusion Timeline ---
    st.subheader("📆 Intrusion Events Over Time")
    intrusions = df[df['Label'] != 'BENIGN']
    if ip_query:
        intrusions = intrusions[intrusions['Source IP'].astype(str).str.contains(ip_query, na=False)]
    time_df = intrusions.dropna(subset=['Timestamp']).groupby(
        intrusions['Timestamp'].dt.floor('H')
    ).size().reset_index(name='Count')
    fig4 = px.line(time_df, x='Timestamp', y='Count')
    st.plotly_chart(fig4, use_container_width=True)

    # --- Search Results Table ---
    if ip_query:
        st.subheader(f"🔍 Search Results for '{ip_query}'")
        result_df = filtered_df[['Timestamp', 'Source IP', 'Protocol', 'Label']]
        st.dataframe(result_df.head(10))

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
                        time.sleep(1)
                    except Exception as e:
                        results[ip] = {"error": str(e)}

                for ip, data in results.items():
                    with st.expander(f"IP: {ip}"):
                        st.json(data)

                # Save JSON for download
                json_str = json.dumps(results, indent=4)
                st.download_button(
                    label="📥 Download Results as JSON",
                    data=json_str,
                    file_name="api_enriched_threats.json",
                    mime="application/json"
                )

    # --- Generate PDF Report ---
    st.subheader("📝 Generate Summary Report (PDF)")

    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Cybersecurity Report", ln=True)

        # Basic stats
        total = len(df)
        malicious = len(df[df['Label'] != 'BENIGN'])
        benign = len(df[df['Label'] == 'BENIGN'])

        pdf.set_font("Arial", '', 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Total Records: {total}", ln=True)
        pdf.cell(0, 10, f"Benign Records: {benign}", ln=True)
        pdf.cell(0, 10, f"Malicious Records: {malicious}", ln=True)

        # Top 5 protocols
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Top 5 Protocols:", ln=True)
        top_protocols = df['Protocol'].value_counts().head(5)
        pdf.set_font("Arial", '', 12)
        for proto, count in top_protocols.items():
            pdf.cell(0, 10, f"{proto}: {count} occurrences", ln=True)

        # Output as BytesIO for Streamlit download
        pdf_output = pdf.output(dest='S').encode('latin1')
        pdf_buffer = BytesIO(pdf_output)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name="cybersecurity_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("👆 Please upload a CSV file to get started.")
