# app_rf_ids.py
import json
import time
import joblib
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

from io import BytesIO
from fpdf import FPDF

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================
API_KEY = "7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4"
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
TOP_N = 30  # number of top IPs to check with AbuseIPDB

MODEL_PATH = "rf_ids_model.joblib"

st.set_page_config(page_title="Threat Intelligence Dashboard", layout="wide")
st.title("Threat Intelligence Dashboard + API Enrichment (Random Forest)")

# =========================
# UTILS
# =========================
EXCLUDE_COLS = {"Label", "Flow ID", "Source IP", "Destination IP", "Timestamp"}

def prepare_xy_for_training(df: pd.DataFrame):
    """Build X, y for training (binary: 0 benign, 1 attack).
       Any Label != 'BENIGN' is considered attack."""
    if "Label" not in df.columns:
        raise ValueError("Training dataset must contain a 'Label' column.")

    y = (df["Label"].astype(str) != "BENIGN").astype(int)

    # timestamp parsing (best-effort)
    if "Timestamp" in df.columns:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # feature candidates = all except excluded
    feature_df = df.drop(columns=[c for c in df.columns if c in EXCLUDE_COLS and c in df.columns])

    # split categorical vs numeric
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    num_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]

    # ColumnTransformer
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ],
        remainder="drop"
    )
    X = feature_df
    meta = {
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "all_feature_cols": feature_df.columns.tolist()
    }
    return X, y, pre, meta

def build_rf_pipeline(preprocessor):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", rf)
    ])
    return pipe

def load_trained_model():
    try:
        obj = joblib.load(MODEL_PATH)
        return obj["pipeline"], obj["meta"]
    except Exception:
        return None, None

def save_trained_model(pipeline, meta):
    joblib.dump({"pipeline": pipeline, "meta": meta}, MODEL_PATH)

def ensure_timestamp(df):
    if "Timestamp" in df.columns:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

def label_available(df):
    return "Label" in df.columns

def add_predictions(df, pipeline, meta):
    """Predict on a dataset without Label (or with, ignoring it).
       Returns dataframe with PredictedLabel (BENIGN/ATTACK) and PredictedClass (0/1)."""
    df = ensure_timestamp(df)
    feat_cols = [c for c in meta["all_feature_cols"] if c in df.columns]
    missing = set(meta["all_feature_cols"]) - set(feat_cols)
    # Create a working copy with all training-time columns; fill missing with NaN
    X = df.copy()
    for m in missing:
        X[m] = pd.NA
    X = X[meta["all_feature_cols"]]

    y_pred = pipeline.predict(X)
    y_pred_label = pd.Series(["BENIGN" if x == 0 else "ATTACK" for x in y_pred], index=df.index, name="PredictedLabel")
    df_out = df.copy()
    df_out["PredictedClass"] = y_pred
    df_out["PredictedLabel"] = y_pred_label
    return df_out

def pie_from_series(series, name_label="Label", name_value="Percentage"):
    rate = series.value_counts(normalize=True).reset_index()
    rate.columns = [name_label, name_value]
    return rate

# Maintain state
if "trained" not in st.session_state:
    st.session_state.trained = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "meta" not in st.session_state:
    st.session_state.meta = None

# Try to load an existing model on startup
if not st.session_state.trained:
    pipe_loaded, meta_loaded = load_trained_model()
    if pipe_loaded is not None:
        st.session_state.pipeline = pipe_loaded
        st.session_state.meta = meta_loaded
        st.session_state.trained = True

# =========================
# LAYOUT: TRAINING PANEL
# =========================
st.header("1) Training (dataset avec Label)")
train_file = st.file_uploader("Importer un dataset CSV pour l'entraînement (doit contenir la colonne 'Label')", type="csv", key="train_uploader")

if train_file is not None:
    df_train = pd.read_csv(train_file)
    st.write(f"Dataset d'entraînement: {df_train.shape[0]} lignes, {df_train.shape[1]} colonnes")
    with st.expander("Aperçu du dataset d'entraînement"):
        st.dataframe(df_train.head(10))

    # Train
    if st.button("Entraîner Random Forest"):
        try:
            X, y, pre, meta = prepare_xy_for_training(df_train)
            pipe = build_rf_pipeline(pre)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            # Cross-validation (optional but informative)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            st.success("Entraînement terminé.")
            st.write(f"Scores de validation croisée (accuracy, 5-fold): {cv_scores.round(4)}")
            st.write(f"Accuracy moyenne CV: {cv_scores.mean():.4f}")

            # Report + confusion matrix
            report = classification_report(y_test, y_pred, output_dict=True)
            st.subheader("Rapport de classification (jeu de test)")
            st.json(report)

            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Benign (0)", "Attack (1)"], columns=["Pred Benign", "Pred Attack"])
            st.subheader("Matrice de confusion (jeu de test)")
            st.dataframe(cm_df)

            # Save model
            save_trained_model(pipe, meta)
            st.session_state.pipeline = pipe
            st.session_state.meta = meta
            st.session_state.trained = True
            st.info("Modèle sauvegardé sur disque et chargé en session.")
        except Exception as e:
            st.error(f"Erreur d'entraînement: {e}")

# =========================
# LAYOUT: INFERENCE PANEL
# =========================
st.header("2) Inference / Dashboard (dataset sans Label accepté)")

uploaded_file = st.file_uploader("Importer un dataset CSV pour l'analyse (avec ou sans 'Label')", type="csv", key="inference_uploader")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = ensure_timestamp(df)

    st.write(f"Dataset pour analyse: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    with st.expander("Aperçu du dataset pour analyse"):
        st.dataframe(df.head(10))

    # If no model available and dataset has Label, allow quick on-the-fly training
    if not st.session_state.trained and label_available(df):
        st.warning("Aucun modèle chargé. La colonne 'Label' est présente : vous pouvez utiliser le panneau Training ci-dessus pour entraîner un modèle.")
    elif not st.session_state.trained and not label_available(df):
        st.error("Aucun modèle chargé et aucune colonne 'Label' dans le dataset. Veuillez entraîner un modèle dans le panneau Training.")
    else:
        # Predict if needed
        if not label_available(df):
            df_pred = add_predictions(df, st.session_state.pipeline, st.session_state.meta)
            st.success("Prédictions effectuées sur dataset sans Label.")
            working_df = df_pred.copy()
            label_col = "PredictedLabel"
            class_col = "PredictedClass"
        else:
            # We have labels; still produce predictions for comparison if user wishes
            df_pred = add_predictions(df, st.session_state.pipeline, st.session_state.meta)
            working_df = df_pred.copy()
            label_col = "Label"
            class_col = "PredictedClass"
            st.info("Le dataset contient 'Label'. Les graphiques 'taux de détection' utiliseront le Label réel, et les IP malveillantes utiliseront les prédictions si besoin.")

        # SEARCH BAR
        st.subheader("Recherche IP / domaine / sous-réseau")
        ip_query = st.text_input("Rechercher dans Source IP (contient)", "")
        if ip_query and "Source IP" in working_df.columns:
            filtered_df = working_df[working_df["Source IP"].astype(str).str.contains(ip_query, na=False)]
        else:
            filtered_df = working_df

        # TRAFFIC BY PROTOCOL (Total vs Malicious)
        st.subheader("Traffic par protocole (Total vs Malicious)")
        # Determine label source: if Label exists, use it; else use PredictedLabel
        label_series = filtered_df[label_col].astype(str) if label_col in filtered_df.columns else pd.Series([], dtype=str)
        if "Protocol" in filtered_df.columns and not label_series.empty:
            grouped = filtered_df.groupby(["Protocol", label_col]).size().reset_index(name="Count")
            fig1 = px.bar(grouped, x="Protocol", y="Count", color=label_col, barmode="group")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Colonnes nécessaires non disponibles pour ce graphique (Protocol et Label/PredictedLabel).")

        # TOP MALICIOUS IPs (use predictions when available)
        st.subheader("Top IPs malveillantes")
        source_col = "Source IP" if "Source IP" in working_df.columns else None
        if source_col:
            if label_col in working_df.columns:
                malicious_mask = working_df[label_col].astype(str) != "BENIGN"
            else:
                malicious_mask = working_df["PredictedLabel"].astype(str) != "BENIGN"

            mal_df = working_df[malicious_mask]
            if ip_query:
                mal_df = mal_df[mal_df[source_col].astype(str).str.contains(ip_query, na=False)]
            if not mal_df.empty:
                top_ips = mal_df[source_col].value_counts().nlargest(10).reset_index()
                top_ips.columns = [source_col, "Count"]
                fig2 = px.bar(top_ips, x=source_col, y="Count")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Aucune IP malveillante détectée sur ce sous-ensemble.")
        else:
            st.info("La colonne 'Source IP' est absente du dataset.")

        # DETECTION RATE PIE CHART
        st.subheader("Taux de détection (Benign vs Malicious)")
        if label_col in filtered_df.columns:
            rate_df = pie_from_series(filtered_df[label_col], name_label="Label", name_value="Percentage")
            fig3 = px.pie(rate_df, values="Percentage", names="Label")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Aucun Label/PredictedLabel pour générer le graphique de taux de détection.")

        # INTRUSION TIME SERIES
        st.subheader("Événements d'intrusion dans le temps")
        if "Timestamp" in working_df.columns and (label_col in working_df.columns):
            intrusions = working_df[working_df[label_col].astype(str) != "BENIGN"].copy()
            if ip_query and source_col:
                intrusions = intrusions[intrusions[source_col].astype(str).str.contains(ip_query, na=False)]
            time_df = intrusions.dropna(subset=["Timestamp"]).groupby(intrusions["Timestamp"].dt.floor("H")).size().reset_index(name="Count")
            if not time_df.empty:
                fig4 = px.line(time_df, x="Timestamp", y="Count")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Pas d'événements d'intrusion à tracer sur la période.")
        else:
            st.info("Colonnes nécessaires non disponibles pour la série temporelle (Timestamp et Label/PredictedLabel).")

        # SEARCH RESULTS TABLE
        if ip_query and source_col:
            st.subheader(f"Résultats de recherche pour '{ip_query}'")
            cols_to_show = [c for c in ["Timestamp", source_col, "Protocol", label_col] if c in filtered_df.columns]
            if cols_to_show:
                st.dataframe(filtered_df[cols_to_show].head(20))
            else:
                st.info("Colonnes suffisantes non disponibles pour afficher un tableau de résultats.")

        # =========================
        # ABUSEIPDB ENRICHMENT
        # =========================
        st.subheader("Enrichissement AbuseIPDB des top IPs malveillantes")
        if not source_col:
            st.error("'Source IP' est nécessaire pour l'enrichissement.")
        else:
            if st.button("Lancer l'IP Reputation Check sur les top IPs malveillantes"):
                if label_col in working_df.columns:
                    mal_mask = working_df[label_col].astype(str) != "BENIGN"
                else:
                    mal_mask = working_df["PredictedLabel"].astype(str) != "BENIGN"

                mal_df = working_df[mal_mask]
                ip_list = mal_df[source_col].value_counts().head(TOP_N).index.tolist()
                results = {}
                with st.spinner("Interrogation AbuseIPDB..."):
                    for ip in ip_list:
                        try:
                            response = requests.get(
                                ABUSEIPDB_URL,
                                headers={"Key": API_KEY, "Accept": "application/json"},
                                params={"ipAddress": ip, "maxAgeInDays": 90, "verbose": True},
                                timeout=15
                            )
                            if response.status_code == 200:
                                results[ip] = response.json()
                            else:
                                results[ip] = {"error": f"Status {response.status_code}", "reason": response.text}
                            time.sleep(1)  # respecter le rate limit
                        except Exception as e:
                            results[ip] = {"error": str(e)}

                for ip, data in results.items():
                    with st.expander(f"IP: {ip}"):
                        st.json(data)

                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Télécharger les résultats d'enrichissement (JSON)",
                    data=json_str,
                    file_name="malicious_ip_enrichment.json",
                    mime="application/json"
                )

        # =========================
        # PDF REPORT
        # =========================
        st.subheader("Générer un rapport PDF")

        def make_pdf_report(df_base, top_ips_df=None, rate_df=None):
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", 'B', 14)
                    self.cell(0, 10, "Cybersecurity Threat Report", ln=True, align="C")
                    self.ln(4)
                def footer(self):
                    self.set_y(-15)
                    self.set_font("Arial", 'I', 8)
                    self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

            pdf = PDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(0, 8, f"Total Records: {len(df_base)}", ln=True)
            label_series_local = df_base[label_col].astype(str) if label_col in df_base.columns else pd.Series([], dtype=str)
            total_malicious = int((label_series_local != "BENIGN").sum()) if not label_series_local.empty else int((df_base["PredictedLabel"] != "BENIGN").sum())
            pdf.cell(0, 8, f"Total Malicious Entries: {total_malicious}", ln=True)

            if top_ips_df is not None and not top_ips_df.empty:
                pdf.ln(4)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Top Malicious IPs:", ln=True)
                pdf.set_font("Arial", size=11)
                for _, row in top_ips_df.head(5).iterrows():
                    pdf.cell(0, 7, f"{row[top_ips_df.columns[0]]}: {row['Count']} detections", ln=True)

            if rate_df is not None and not rate_df.empty:
                pdf.ln(4)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Detection Breakdown:", ln=True)
                pdf.set_font("Arial", size=11)
                for _, r in rate_df.iterrows():
                    pdf.cell(0, 7, f"{r['Label']}: {round(r['Percentage']*100, 2)}%", ln=True)

            out_bytes = BytesIO()
            pdf.output(out_bytes)
            return out_bytes.getvalue()

        # Compute top_ips and rate to include in PDF
        top_ips_pdf = None
        if source_col:
            if label_col in working_df.columns:
                mal_mask_pdf = working_df[label_col].astype(str) != "BENIGN"
            else:
                mal_mask_pdf = working_df["PredictedLabel"].astype(str) != "BENIGN"
            mal_df_pdf = working_df[mal_mask_pdf]
            if not mal_df_pdf.empty:
                top_ips_pdf = mal_df_pdf[source_col].value_counts().nlargest(10).reset_index()
                top_ips_pdf.columns = [source_col, "Count"]

        rate_pdf = None
        if label_col in working_df.columns:
            rate_pdf = pie_from_series(working_df[label_col], "Label", "Percentage")

        if st.button("Générer le PDF"):
            pdf_bytes = make_pdf_report(working_df, top_ips_pdf, rate_pdf)
            st.download_button(
                "Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name="threat_report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Importez un dataset CSV pour démarrer l'analyse.")
