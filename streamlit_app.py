import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests
import json
import time
from fpdf import FPDF

# === AJOUTS POUR L'AUTO-LABELING RANDOM FOREST (ne modifient pas l'UI) ===
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# === CONFIG ===
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"  # <-- Replace with your AbuseIPDB API key
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
TOP_N = 30  # Number of top IPs to check

# === MODELES (chemins d'artefacts) ===
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
SCALER_PATH = MODEL_DIR / "rf_scaler.pkl"
FEATS_PATH = MODEL_DIR / "rf_features.json"

# Colonnes à exclure des features du modèle
_EXCLUDE_COLS = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"}

def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne des colonnes numériques exploitables, exclut identifiants et nettoie NaN/inf/constantes."""
    num = df.select_dtypes(include=[np.number]).copy()
    for c in _EXCLUDE_COLS:
        if c in num.columns:
            num.drop(columns=[c], inplace=True, errors='ignore')
    # Supprimer colonnes constantes
    nunique = num.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    num = num[keep]
    # Nettoyage NaN/inf
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0)
    return num

def _load_model_bundle():
    if MODEL_PATH.exists() and SCALER_PATH.exists() and FEATS_PATH.exists():
        try:
            rf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            with open(FEATS_PATH, "r", encoding="utf-8") as f:
                feats = json.load(f)
            return rf, scaler, feats
        except Exception:
            return None, None, None
    return None, None, None

def _save_model_bundle(rf, scaler, feats):
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(feats), f, indent=2)

def ensure_labels_rf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit la présence de df['Label'] :
      - Si 'Label' existe : renvoie df tel quel (UI non modifiée).
      - Sinon :
          * tente de charger un modèle RF pré-entraîné (features alignées),
          * à défaut, réalise un pseudo-labeling KMeans(n_clusters=2),
            entraîne un RandomForestClassifier, prédit et assigne 'Label' ∈ {BENIGN, ATTACK},
            puis sauvegarde le modèle pour réutilisation future.
      - Si aucune feature exploitable : assigne 'BENIGN' par défaut (sécurité UI).
    """
    if "Label" in df.columns:
        return df

    X = _select_numeric_features(df)
    df = df.copy()

    if not X.empty:
        # 1) Essayer un modèle déjà entraîné
        rf, scaler, feats = _load_model_bundle()
        if rf is not None and scaler is not None and feats is not None:
            # réaligner les features (ajout de colonnes manquantes à 0)
            for col in feats:
                if col not in X.columns:
                    X[col] = 0
            X_aligned = X[feats]
            Xs = scaler.transform(X_aligned)
            y_pred = rf.predict(Xs)
            df["Label"] = np.where(y_pred == 1, "ATTACK", "BENIGN")
            return df

        # 2) Pas de modèle : pseudo-labeling + entraînement RF
        scaler_pl = StandardScaler()
        Xs = scaler_pl.fit_transform(X)

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = km.fit_predict(Xs)

        # Heuristique : cluster majoritaire = BENIGN
        counts = pd.Series(clusters).value_counts().sort_values(ascending=False)
        benign_cluster = counts.index[0]
        y_pseudo = np.where(clusters == benign_cluster, 0, 1)  # 0=BENIGN, 1=ATTACK

        rf_pl = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample"
        )
        rf_pl.fit(Xs, y_pseudo)

        y_hat = rf_pl.predict(Xs)
        df["Label"] = np.where(y_hat == 1, "ATTACK", "BENIGN")

        # Sauvegarder artefacts pour prochaines prédictions
        _save_model_bundle(rf_pl, scaler_pl, X.columns)
        return df

    # 3) Pas de features numériques exploitables
    df["Label"] = "BENIGN"
    return df

# === PAGE SETTINGS ===
st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")
st.title("🔐 Threat Intelligence Dashboard + API Enrichment")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # === AJOUT : garantir la présence de 'Label' par Random Forest si absent ===
    df = ensure_labels_rf(df)

    # === SEARCH BAR ===
    ip_query = st.text_input("🔎 Search IP/domain/subnet", "")
    filtered_df = df[df['Source IP'].astype(str).str.contains(ip_query, na=False)] if ip_query else df

    # === TRAFFIC BY PROTOCOL ===
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
        total_malicious = len(df[df['Label'] != 'BENIGN'])
        pdf.cell(0, 10, f"Total Malicious Entries: {total_malicious}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Top 5 Malicious IPs:", ln=True)
        pdf.set_font("Arial", size=11)
        for idx, row in top_ips.head(5).iterrows():
            pdf.cell(0, 10, f"{row['Source IP']}: {row['Count']} detections", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Detection Breakdown:", ln=True)
        pdf.set_font("Arial", size=11)
        for i, r in rate.iterrows():
            pdf.cell(0, 10, f"{r['Label']}: {round(r['Percentage']*100, 2)}%", ln=True)

        pdf_output = pdf.output(dest='S').encode('latin1')
        st.download_button("📄 Download PDF Report", data=pdf_output, file_name="threat_report.pdf", mime="application/pdf")

else:
    st.info("👆 Please upload a CSV file to begin.")
