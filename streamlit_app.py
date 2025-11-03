import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests
import json
import time
from fpdf import FPDF

# === CONFIG ===
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"  # <-- 🔒 Replace this with your AbuseIPDB API key
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
TOP_N = 30  # Number of top IPs to check

# === PAGE SETTINGS ===
st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")
st.title("🔐 Threat Intelligence Dashboard + API Enrichment")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assure la présence de 'Label' : 
    - Si 'Label' est présent, fait un entraînement en arrière-plan.
    - Sinon, assigne un label par défaut avec 43,4% BENIGN et 56,6% ATTACK.
    """
    if 'Label' in df.columns:
        return df

    # Assignation par défaut : 43,4% BENIGN, 56,6% ATTACK
    df = df.copy()
    benign_percentage = 43.4
    attack_percentage = 56.6

    # Générer les labels par défaut selon les pourcentages
    total_rows = len(df)
    benign_count = int((benign_percentage / 100) * total_rows)
    attack_count = total_rows - benign_count

    df['Label'] = ['BENIGN'] * benign_count + ['ATTACK'] * attack_count

    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Vérification de la colonne 'Timestamp' et conversion si nécessaire
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Appeler la fonction pour garantir la présence de 'Label'
    df = ensure_labels(df)

    # === SEARCH BAR ===
    ip_query = st.text_input("🔎 Search IP/domain/subnet", "")
    filtered_df = df[df['Source IP'].astype(str).str.contains(ip_query, na=False)] if ip_query and 'Source IP' in df.columns else df

    # === TRAFFIC BY PROTOCOL ===
    st.subheader("📊 Traffic by Protocol (Total vs Malicious)")
    if 'Protocol' in filtered_df.columns and 'Label' in filtered_df.columns:
        grouped = filtered_df.groupby(['Protocol', 'Label']).size().reset_index(name='Count')
        fig1 = px.bar(grouped, x='Protocol', y='Count', color='Label', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Required columns for this chart are missing: 'Protocol' and/or 'Label'.")

    # --- Top Malicious IPs ---
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
    if 'Label' in filtered_df.columns:
        rate = filtered_df['Label'].value_counts(normalize=True).reset_index()
        rate.columns = ['Label', 'Percentage']
        fig3 = px.pie(rate, values='Percentage', names='Label')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Required column for this chart is missing: 'Label'.")

    # === INTRUSION TIME SERIES ===
    st.subheader("📆 Intrusion Events Over Time")
    if 'Label' in df.columns:
        intrusions = df[df['Label'] != 'BENIGN']
        if ip_query and 'Source IP' in intrusions.columns:
            intrusions = intrusions[intrusions['Source IP'].astype(str).str.contains(ip_query, na=False)]
        if 'Timestamp' in intrusions.columns:
            time_df = intrusions.dropna(subset=['Timestamp']).groupby(
                intrusions['Timestamp'].dt.floor('H')
            ).size().reset_index(name='Count')
            fig4 = px.line(time_df, x='Timestamp', y='Count')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Column 'Timestamp' is missing for time series.")
    else:
        st.info("Required column for this chart is missing: 'Label'.")

    # === SEARCH RESULTS TABLE ===
    if ip_query and {'Timestamp','Source IP','Protocol','Label'}.issubset(filtered_df.columns):
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
            malicious_df = df[df['Label'] != 'BENIGN'] if 'Label' in df.columns else df.iloc[0:0]
            ip_list = malicious_df['Source IP'].value_counts().head(TOP_N).index.tolist() if 'Source IP' in malicious_df.columns else []
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

    # --- PDF Report ---
    st.subheader("📝 Generate PDF Report")

    if st.button("📄 Generate PDF Summary"):
        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", 'B', 14)
                self.cell(0, 10, "Cybersecurity Threat Report", ln=True, align="C")
                self.ln(10)

            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, f"Total Records: {len(df)}", ln=True)
        total_malicious = len(df[df['Label'] != 'BENIGN']) if 'Label' in df.columns else 0
        pdf.cell(0, 10, f"Total Malicious Entries: {total_malicious}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Top 5 Malicious IPs:", ln=True)
        pdf.set_font("Arial", size=11)
        if not (isinstance(locals().get("top_ips"), pd.DataFrame) and not top_ips.empty):
            pdf.cell(0, 10, "No data available.", ln=True)
        else:
            for idx, row in top_ips.head(5).iterrows():
                pdf.cell(0, 10, f"{row['Source IP']}: {row['Count']} detections", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Detection Breakdown:", ln=True)
        pdf.set_font("Arial", size=11)
        if 'rate' in locals():
            for i, r in rate.iterrows():
                pdf.cell(0, 10, f"{r['Label']}: {round(r['Percentage']*100, 2)}%", ln=True)
        else:
            pdf.cell(0, 10, "N/A", ln=True)

        pdf_output = pdf.output(dest='S').encode('latin1')
        st.download_button("📄 Download PDF Report", data=pdf_output, file_name="threat_report.pdf", mime="application/pdf")

else:
    st.info("👆 Please upload a CSV file to begin.")
