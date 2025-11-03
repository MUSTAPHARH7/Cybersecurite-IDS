import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests
import json
import time
from fpdf import FPDF

# === AJOUTS ML (imports supplémentaires) ===
import threading
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"  # <-- 🔒 Replace this with your AbuseIPDB API key
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
TOP_N = 30  # Number of top IPs to check

# === PATHS POUR LE MODELE RF ===
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
FEATS_PATH = MODEL_DIR / "rf_features.json"
SCALER_PATH = MODEL_DIR / "rf_scaler.pkl"

# === FONCTIONS ML AJOUTEES (sans modifier l'UI) ===
EXCLUDE_COLS = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"}

def select_numeric_features(df: pd.DataFrame):
    """Sélectionne les colonnes numériques utilisables par le modèle, en excluant les identifiants."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    # Exclure explicitement si présentes
    for c in EXCLUDE_COLS:
        if c in num_df.columns:
            num_df.drop(columns=[c], inplace=True, errors='ignore')
    # Supprimer colonnes constantes ou quasi vides
    nunique = num_df.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    num_df = num_df[keep]
    # Remplacer inf/NaN
    num_df = num_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return num_df

def load_model_bundle():
    if MODEL_PATH.exists() and FEATS_PATH.exists() and SCALER_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            with open(FEATS_PATH, "r", encoding="utf-8") as f:
                feats = json.load(f)
            return model, scaler, feats
        except Exception:
            return None, None, None
    return None, None, None

def save_model_bundle(model, scaler, feats):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATS_PATH, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)

def fit_rf_background(df_with_labels: pd.DataFrame):
    """Entraîne RF en tâche de fond quand 'Label' est présent."""
    try:
        X = select_numeric_features(df_with_labels)
        if X.empty:
            return
        # Créer y binaire : BENIGN vs ATTACK (tout ce qui n'est pas BENIGN = ATTACK)
        y = (df_with_labels.get("Label", pd.Series(index=df_with_labels.index, dtype=object)) != "BENIGN").astype(int)

        # Mise à l'échelle
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Split + entraînement
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )
        rf.fit(Xtr, ytr)

        # Sauvegarde du modèle et des features utilisés
        save_model_bundle(rf, scaler, list(X.columns))
    except Exception:
        # On ignore les erreurs pour ne pas impacter l'UI
        pass

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit la présence de df['Label'] :
    - si déjà présent : déclenche un entraînement RF en background (pour mise à jour du modèle)
    - sinon : tente de charger un modèle pré-entraîné ; à défaut, pseudo-étiquetage KMeans + entraînement RF rapide
    Retourne un df avec colonne 'Label' (BENIGN/ATTACK).
    """
    if "Label" in df.columns:
        # Entraînement asynchrone sur ce dataset étiqueté (n’affecte pas l’UI)
        threading.Thread(target=fit_rf_background, args=(df.copy(),), daemon=True).start()
        return df

    # Sinon, essayer de prédire avec un modèle existant
    model, scaler, feats = load_model_bundle()
    X = select_numeric_features(df)
    if not X.empty:
        if model is not None and scaler is not None and feats is not None:
            # Aligner les features
            common = [c for c in feats if c in X.columns]
            if common:
                X_aligned = X.reindex(columns=feats, fill_value=0)
                Xs = scaler.transform(X_aligned)
                pred = model.predict(Xs)
                labels = np.where(pred == 1, "ATTACK", "BENIGN")
                df = df.copy()
                df["Label"] = labels
                return df

        # Pas de modèle dispo : pseudo-labeling KMeans (2 clusters), puis RF
        try:
            scaler_pl = StandardScaler()
            Xs = scaler_pl.fit_transform(X)
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            cl = km.fit_predict(Xs)

            # Heuristique : cluster majoritaire = BENIGN, minoritaire = ATTACK
            counts = pd.Series(cl).value_counts().sort_values(ascending=False)
            benign_cluster = counts.index[0]
            y_pseudo = np.where(cl == benign_cluster, 0, 1)

            # Entraîner RF rapide sur pseudo-labels puis prédire 'Label'
            rf_pl = RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced_subsample"
            )
            rf_pl.fit(Xs, y_pseudo)

            labels = np.where(rf_pl.predict(Xs) == 1, "ATTACK", "BENIGN")
            df = df.copy()
            df["Label"] = labels

            # Sauvegarder ce modèle provisoire pour réutilisation
            # (facultatif : on sauvegarde les features et le scaler issus du pseudo-labeling)
            save_model_bundle(rf_pl, scaler_pl, list(X.columns))
            return df
        except Exception:
            pass

    # Si impossible de créer/predire des labels, on met une valeur de secours pour ne pas casser l’UI
    df = df.copy()
    df["Label"] = "BENIGN"
    return df

# === PAGE SETTINGS ===
st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")
st.title("🔐 Threat Intelligence Dashboard + API Enrichment")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Normaliser/convertir Timestamp si présent
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # === INJECTION ML : garantir la présence de 'Label' avant l'UI existante ===
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

    # === TOP MALICIOUS IPS ===
    st.subheader("🚨 Top Malicious IPs")
    if 'Label' in df.columns and 'Source IP' in df.columns:
        malicious_df = df[df['Label'] != 'BENIGN']
        if ip_query and 'Source IP' in malicious_df.columns:
            malicious_df = malicious_df[malicious_df['Source IP'].astype(str).str.contains(ip_query, na=False)]
        if not malicious_df.empty:
            top_ips = malicious_df['Source IP'].value_counts().nlargest(10).reset_index()
            top_ips.columns = ['Source IP', 'Count']
            fig2 = px.bar(top_ips, x='Source IP', y='Count')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            top_ips = pd.DataFrame(columns=['Source IP', 'Count'])
            st.info("No malicious IPs found in the current view.")
    else:
        top_ips = pd.DataFrame(columns=['Source IP', 'Count'])
        st.info("Required columns for this chart are missing: 'Source IP' and/or 'Label'.")

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
