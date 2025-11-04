import os
import re
import json
import time
import joblib
import requests
import pandas as pd
import plotly.express as px

from functools import lru_cache
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from dash import Dash, dcc, html, Input, Output, State, callback_context

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = "dashboard_data.csv"
MODEL_PATH = "rf_model.pkl"          # modèle RandomForest pour prédire Label si absent
ABUSEIPDB_KEY = os.getenv("7fd4c5eba9c28f0b846f1f8e3ae013380bf4af60ec50f865d0163d2431b9bd8474caef849e8393a4")  # ta clé AbuseIPDB (exporte-la dans l'env)
ABUSE_MAX_AGE_DAYS = 90              # fenêtre d'analyse chez AbuseIPDB
TOP_K_ENRICH = 10                    # nombre d'IP à enrichir automatiquement dans le top

# ============================================================================
# DATA LOADING & LABEL HANDLING
# ============================================================================

ATTACK_ALIASES = {"ATTACK", "MALICIOUS", "MALWARE", "ANOMALY", "INTRUSION", "ABNORMAL", "DDOS", "DOS", "BOTNET"}
BENIGN_ALIASES = {"BENIGN", "NORMAL", "CLEAN", "LEGIT", "SAFE"}

def normalize_label_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in {"i", "u", "f"}:
        return (s.astype(float) > 0).astype(int)

    def map_one(x):
        if pd.isna(x):
            return None
        xs = str(x).strip().upper()
        if xs in BENIGN_ALIASES:
            return 0
        if xs in ATTACK_ALIASES:
            return 1
        if any(k in xs for k in ["ATTACK", "MALIC", "ANOM", "INTRU", "DDOS", "DOS", "BOT", "PORTSCAN"]):
            return 1
        if any(k in xs for k in ["BENIGN", "NORMAL", "CLEAN", "LEGIT", "SAFE"]):
            return 0
        return None

    mapped = s.map(map_one)
    if mapped.isna().any():
        fill_val = mapped.mode().iloc[0] if not mapped.dropna().empty else 0
        mapped = mapped.fillna(fill_val)
    return mapped.astype(int)

def build_pipeline(numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    return Pipeline(steps=[("prep", preprocessor), ("rf", rf)])

def ensure_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    if "Label" in df.columns:
        df["Label"] = normalize_label_series(df["Label"])
        return df

    # Pas de Label -> on tente de charger un modèle
    if os.path.exists(MODEL_PATH):
        # Déterminer types de colonnes pour le pipeline à partir du df courant
        ignore = {"Label", "Timestamp"}
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in ignore]
        categorical_cols = [c for c in df.columns if c not in ignore and c not in numeric_cols]

        try:
            art = joblib.load(MODEL_PATH)
            if isinstance(art, dict) and "pipeline" in art:
                clf = art["pipeline"]
            else:
                # compat: si le joblib contient directement le pipeline
                clf = art
        except Exception:
            clf = None

        if clf is None:
            # modèle indisponible ou corrompu -> on crée à la volée un pipeline vide pour lever une erreur claire
            raise RuntimeError("Impossible de charger le modèle RandomForest. Recréez rf_model.pkl.")

        # Si le pipeline a été entraîné sur d'autres colonnes, il sait ignorer l'inconnu (OneHot handle_unknown)
        # On prédit
        try:
            preds = clf.predict(df)
            df["Label"] = preds.astype(int)
        except Exception as e:
            raise RuntimeError(f"Le modèle chargé ne peut pas prédire sur ce dataset: {e}")
        return df

    # Aucun Label et aucun modèle
    raise RuntimeError(
        "La colonne 'Label' est absente et aucun modèle pré-entraîné 'rf_model.pkl' n'a été trouvé."
        " Entraînez un modèle sur un dataset labellisé, puis relancez."
    )

def load_df():
    base = pd.read_csv(DATA_PATH)
    return ensure_label(base)

# ============================================================================
# ABUSEIPDB INTEGRATION
# ============================================================================

IPV4_REGEX = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})$")

def is_ipv4(x: str) -> bool:
    return bool(IPV4_REGEX.match(x or ""))

@lru_cache(maxsize=2048)
def abuseipdb_check_cached(ip: str, max_age_days: int = ABUSE_MAX_AGE_DAYS) -> dict:
    """Cache simple mémoire pour limiter les appels/rate-limit."""
    return abuseipdb_check(ip, max_age_days=max_age_days)

def abuseipdb_check(ip: str, max_age_days: int = ABUSE_MAX_AGE_DAYS) -> dict:
    if not ABUSEIPDB_KEY:
        return {"error": "ABUSEIPDB_KEY not set in environment."}
    url = "https://api.abuseipdb.com/api/v2/check"
    headers = {"Key": ABUSEIPDB_KEY, "Accept": "application/json"}
    params = {"ipAddress": ip, "maxAgeInDays": max_age_days, "verbose": True}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 429:
            # rate limit -> petit backoff
            time.sleep(1.5)
            r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", {})
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}

def enrich_ips_with_abuse(ip_list):
    out = []
    for ip in ip_list:
        if is_ipv4(ip):
            res = abuseipdb_check_cached(ip)
            score = res.get("abuseConfidenceScore")
            total = res.get("totalReports")
            country = res.get("countryCode")
            usage = (res.get("usageType") or "")[:40]
            out.append({"Source IP": ip, "abuseConfidenceScore": score, "totalReports": total,
                        "country": country, "usageType": usage})
        else:
            out.append({"Source IP": ip, "abuseConfidenceScore": None, "totalReports": None,
                        "country": None, "usageType": None})
    return pd.DataFrame(out)

# ============================================================================
# DASH APP
# ============================================================================

app = Dash(__name__)
app.title = "Cybersecurity Dashboard"

app.layout = html.Div([
    html.H1("Threat Intelligence Dashboard"),

    # Search bar
    dcc.Input(
        id="ip-search",
        type="text",
        placeholder="Search IP/domain/subnet",
        debounce=True,
        style={'marginBottom': '20px', 'width': '50%'}
    ),

    # Hidden store for periodic data refresh + enrichment cache
    dcc.Store(id="data-store"),
    dcc.Store(id="top-enriched-store"),

    # auto-refresh (5 min)
    dcc.Interval(id="refresh", interval=5*60*1000, n_intervals=0),

    # Graphs
    dcc.Graph(id="traffic-by-protocol"),
    dcc.Graph(id="top-malicious-ips"),
    dcc.Graph(id="detection-rate"),
    dcc.Graph(id="intrusion-time-series"),

    # AbuseIPDB Intel (for search)
    html.Div(id="abuse-intel", style={"marginTop": "10px"}),

    # Search Results Section
    html.Div(id='search-results')
])

# ============================================================================
# Shared data: periodic reload
# ============================================================================

@app.callback(
    Output("data-store", "data"),
    Input("refresh", "n_intervals")
)
def refresh_data(_):
    try:
        df = load_df()
        return df.to_json(date_format="iso", orient="records")
    except Exception as e:
        # on renvoie None -> les callbacks consommateurs gèrent
        return json.dumps({"__error__": str(e)})

def get_df_from_store(data_json):
    if not data_json:
        return load_df()  # premier run
    try:
        df = pd.DataFrame(json.loads(data_json))
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df
    except Exception:
        return load_df()

# ============================================================================
# Traffic by Protocol
# ============================================================================

@app.callback(
    Output('traffic-by-protocol', 'figure'),
    Input('ip-search', 'value'),
    State("data-store", "data")
)
def update_traffic_by_protocol(ip_query, data_json):
    df = get_df_from_store(data_json)
    filtered_df = df if not ip_query else df[df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    grouped = filtered_df.groupby(['Protocol', 'Label']).size().reset_index(name='Count')
    return px.bar(
        grouped, x='Protocol', y='Count', color='Label', barmode='group',
        title='Traffic by Protocol (Total vs Malicious)'
    )

# ============================================================================
# Top Malicious IPs (+ auto-enrichment)
# ============================================================================

@app.callback(
    Output('top-malicious-ips', 'figure'),
    Output('top-enriched-store', 'data'),
    Input('ip-search', 'value'),
    State("data-store", "data")
)
def update_top_ips(ip_query, data_json):
    df = get_df_from_store(data_json)
    filtered_df = df[df['Label'] != 0] if df['Label'].dtype != 'O' else df[df['Label'] != 'BENIGN']
    if ip_query:
        filtered_df = filtered_df[filtered_df['Source IP'].astype(str).str.contains(ip_query, na=False)]

    top_ips = filtered_df['Source IP'].value_counts().nlargest(10).reset_index()
    top_ips.columns = ['Source IP', 'Count']

    # Enrichissement AbuseIPDB automatique (top K du full malicieux, pas seulement du filtre)
    try:
        overall_mal = df[df['Label'] == 1] if df['Label'].dtype != 'O' else df[df['Label'] != 'BENIGN']
        overall_top = overall_mal['Source IP'].value_counts().nlargest(TOP_K_ENRICH).index.tolist()
        enriched = enrich_ips_with_abuse(overall_top)
        enriched_json = enriched.to_json(orient="records")
    except Exception:
        enriched_json = None

    fig = px.bar(top_ips, x='Source IP', y='Count', title='Top Malicious IPs')
    return fig, enriched_json

# ============================================================================
# Detection Rate Pie Chart
# ============================================================================

@app.callback(
    Output('detection-rate', 'figure'),
    Input('ip-search', 'value'),
    State("data-store", "data")
)
def update_detection_rate(ip_query, data_json):
    df = get_df_from_store(data_json)
    filtered_df = df if not ip_query else df[df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    rate = filtered_df['Label'].value_counts(normalize=True).reset_index()
    rate.columns = ['Label', 'Percentage']
    # Affichage human-friendly si binaire
    if rate['Label'].dtype != 'O':
        rate['Label'] = rate['Label'].map({0: "BENIGN (0)", 1: "ATTACK (1)"}).fillna(rate['Label'])
    return px.pie(
        rate, values='Percentage', names='Label', title='Detection Rate (Benign vs Malicious)'
    )

# ============================================================================
# Time-Series Intrusion Chart
# ============================================================================

@app.callback(
    Output('intrusion-time-series', 'figure'),
    Input('ip-search', 'value'),
    State("data-store", "data")
)
def update_time_series(ip_query, data_json):
    df = get_df_from_store(data_json)
    filtered_df = df[df['Label'] == 1] if df['Label'].dtype != 'O' else df[df['Label'] != 'BENIGN']
    if ip_query:
        filtered_df = filtered_df[filtered_df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    time_df = filtered_df.dropna(subset=['Timestamp']).groupby(
        filtered_df['Timestamp'].dt.floor('H')
    ).size().reset_index(name='Count')
    return px.line(
        time_df, x='Timestamp', y='Count', title='Intrusion Events Over Time'
    )

# ============================================================================
# AbuseIPDB Intel for searched IP + Search Results table
# ============================================================================

@app.callback(
    Output('abuse-intel', 'children'),
    Output('search-results', 'children'),
    Input('ip-search', 'value'),
    State("data-store", "data"),
    State("top-enriched-store", "data")
)
def display_search_results(ip_query, data_json, enriched_json):
    df = get_df_from_store(data_json)

    # --- Search results table (top 10)
    if ip_query:
        matches = df[df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    else:
        matches = df.iloc[0:0]  # empty

    rows = []
    for i in range(min(len(matches), 10)):
        rows.append(html.Tr([
            html.Td(str(matches.iloc[i]['Timestamp'])),
            html.Td(matches.iloc[i]['Source IP']),
            html.Td(matches.iloc[i]['Protocol']),
            html.Td(matches.iloc[i]['Label'])
        ]))

    table = html.Div([
        html.H4(f"Search Results for '{ip_query}': {len(matches)} matching entries") if ip_query else html.P(""),
        html.Table(
            [html.Tr([html.Th(c) for c in ['Timestamp', 'Source IP', 'Protocol', 'Label']])] + rows
        )
    ])

    # --- AbuseIPDB intel box (only if query is a single IPv4)
    intel_box = html.Div()
    if ip_query and is_ipv4(ip_query):
        # AbuseIPDB lookup
        info = abuseipdb_check_cached(ip_query)
        if "error" in info:
            intel_box = html.Div([
                html.H4("AbuseIPDB Intel"),
                html.P(info["error"])
            ], style={"marginTop": "10px"})
        else:
            intel_box = html.Div([
                html.H4("AbuseIPDB Intel"),
                html.Ul([
                    html.Li(f"IP: {info.get('ipAddress')}"),
                    html.Li(f"Country: {info.get('countryCode')}"),
                    html.Li(f"Usage: {info.get('usageType')}"),
                    html.Li(f"Domain: {info.get('domain')}"),
                    html.Li(f"ISP: {info.get('isp')}"),
                    html.Li(f"Abuse Confidence Score: {info.get('abuseConfidenceScore')}"),
                    html.Li(f"Total Reports: {info.get('totalReports')}"),
                    html.Li(f"Last Reported: {info.get('lastReportedAt')}")
                ])
            ], style={"marginTop": "10px"})

    # --- Optional: show enriched top malicious summary under the table
    enrich_block = html.Div()
    if enriched_json:
        try:
            enr = pd.DataFrame(json.loads(enriched_json))
            if not enr.empty:
                # petite table HTML
                header = html.Tr([html.Th(c) for c in ["Source IP", "abuseConfidenceScore", "totalReports", "country", "usageType"]])
                rows = [
                    html.Tr([html.Td(str(row[c])) for c in ["Source IP", "abuseConfidenceScore", "totalReports", "country", "usageType"]])
                    for _, row in enr.iterrows()
                ]
                enrich_block = html.Div([
                    html.H4("AbuseIPDB Enrichment (Top Malicious IPs)"),
                    html.Table([header] + rows)
                ], style={"marginTop": "16px"})
        except Exception:
            pass

    return intel_box, html.Div([table, enrich_block])

# ============================================================================
# Run the server
# ============================================================================
if __name__ == '__main__':
    # Premier chargement de df -> crée le cache au démarrage (et vérifie le modèle)
    try:
        _ = load_df()
    except Exception as e:
        print(f"[WARN] Data/model init: {e}")
    app.run(debug=True)
