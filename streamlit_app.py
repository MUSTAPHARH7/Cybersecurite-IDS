import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Load dataset
df = pd.read_csv('dashboard_data.csv')

# Preprocess Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Create the Dash app
app = Dash(__name__)
app.title = "Cybersecurity Dashboard"

# Layout definition
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

    # Graphs
    dcc.Graph(id="traffic-by-protocol"),
    dcc.Graph(id="top-malicious-ips"),
    dcc.Graph(id="detection-rate"),
    dcc.Graph(id="intrusion-time-series"),

    # Search Results Section
    html.Div(id='search-results')
])

# Traffic by Protocol
@app.callback(
    Output('traffic-by-protocol', 'figure'),
    Input('ip-search', 'value')
)
def update_traffic_by_protocol(ip_query):
    filtered_df = df if not ip_query else df[df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    grouped = filtered_df.groupby(['Protocol', 'Label']).size().reset_index(name='Count')
    return px.bar(
        grouped, x='Protocol', y='Count', color='Label', barmode='group',
        title='Traffic by Protocol (Total vs Malicious)'
    )

# Top Malicious IPs
@app.callback(
    Output('top-malicious-ips', 'figure'),
    Input('ip-search', 'value')
)
def update_top_ips(ip_query):
    filtered_df = df[df['Label'] != 'BENIGN']
    if ip_query:
        filtered_df = filtered_df[filtered_df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    top_ips = filtered_df['Source IP'].value_counts().nlargest(10).reset_index()
    top_ips.columns = ['Source IP', 'Count']
    return px.bar(
        top_ips, x='Source IP', y='Count', title='Top Malicious IPs'
    )

# Detection Rate Pie Chart
@app.callback(
    Output('detection-rate', 'figure'),
    Input('ip-search', 'value')
)
def update_detection_rate(ip_query):
    filtered_df = df if not ip_query else df[df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    rate = filtered_df['Label'].value_counts(normalize=True).reset_index()
    rate.columns = ['Label', 'Percentage']
    return px.pie(
        rate, values='Percentage', names='Label', title='Detection Rate (Benign vs Malicious)'
    )

# Time-Series Intrusion Chart
@app.callback(
    Output('intrusion-time-series', 'figure'),
    Input('ip-search', 'value')
)
def update_time_series(ip_query):
    filtered_df = df[df['Label'] != 'BENIGN']
    if ip_query:
        filtered_df = filtered_df[filtered_df['Source IP'].astype(str).str.contains(ip_query, na=False)]
    time_df = filtered_df.dropna(subset=['Timestamp']).groupby(
        filtered_df['Timestamp'].dt.floor('H')
    ).size().reset_index(name='Count')
    return px.line(
        time_df, x='Timestamp', y='Count', title='Intrusion Events Over Time'
    )

# Search Results Display
@app.callback(
    Output('search-results', 'children'),
    Input('ip-search', 'value')
)
def display_search_results(ip_query):
    if not ip_query:
        return html.P("")

    matches = df[df['Source IP'].astype(str).str.contains(ip_query, na=False)]

    return html.Div([
        html.H4(f"Search Results for '{ip_query}': {len(matches)} matching entries"),
        html.Table([
            html.Tr([html.Th(col) for col in ['Timestamp', 'Source IP', 'Protocol', 'Label']])
        ] + [
            html.Tr([
                html.Td(str(matches.iloc[i]['Timestamp'])),
                html.Td(matches.iloc[i]['Source IP']),
                html.Td(matches.iloc[i]['Protocol']),
                html.Td(matches.iloc[i]['Label'])
            ]) for i in range(min(len(matches), 10))  # limit to first 10 matches
        ])
    ])

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
