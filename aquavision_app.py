"""
💧 AquaVision — Water Quality Intelligence
Final Year Project | Streamlit Dashboard App with Bulk Scanner
Run: streamlit run aquavision_app_enhanced.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
import io
import csv
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import xgboost as xgb

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaVision | Water Quality Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Use system fonts for guaranteed rendering */
    html, body, [class*="css"], .stMarkdown, .stText, p, div, span, h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    .main { background: #F0F8FF; }
    
    .hero-banner {
        background: linear-gradient(135deg, #0D47A1 0%, #006064 50%, #1B5E20 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(13,71,161,0.3);
    }
    .hero-banner h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: white !important;
        margin: 0;
        letter-spacing: -1px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .hero-banner p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9) !important;
        margin: 0.5rem 0 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #1565C0;
        margin-bottom: 0.8rem;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #555 !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: #0D47A1 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #777 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .wqi-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 700;
        color: white !important;
        margin-top: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .predict-box {
        background: linear-gradient(135deg, #E3F2FD, #E8F5E9);
        border-radius: 16px;
        padding: 2rem;
        border: 2px solid #90CAF9;
        box-shadow: 0 4px 20px rgba(21,101,192,0.15);
        margin-bottom: 1.5rem;
    }
    
    /* Section title - highly visible */
    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0D47A1 !important;
        margin: 1.5rem 0 1rem;
        padding: 0.6rem 0.8rem;
        border-left: 5px solid #1565C0;
        background: #EEF4FF;
        border-radius: 0 8px 8px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }

    /* ── SIDEBAR — ALL TEXT WHITE & HIGHLY VISIBLE ─────────────────────────── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D47A1 0%, #006064 100%) !important;
    }
    
    /* Force all sidebar text to be bright white */
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] h4,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Selectbox styling in sidebar */
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stNumberInput label,
    div[data-testid="stSidebar"] .stTextInput label {
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    
    /* Selectbox dropdown styling */
    div[data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: rgba(255,255,255,0.12) !important;
    }
    
    div[data-testid="stSidebar"] [data-baseweb="select"] span,
    div[data-testid="stSidebar"] [data-baseweb="select"] div,
    div[data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: white !important;
    }
    
    div[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.15) !important;
        border: 1.5px solid rgba(255,255,255,0.4) !important;
        color: white !important;
    }
    
    /* Slider styling */
    div[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    div[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: white !important;
    }
    
    /* Divider/HR styling */
    div[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.3) !important;
    }
    
    /* Markdown text in sidebar */
    div[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }
    
    /* Input field styling in sidebar */
    div[data-testid="stSidebar"] input,
    div[data-testid="stSidebar"] textarea {
        color: white !important;
        background-color: rgba(255,255,255,0.1) !important;
    }
    
    /* Ensure filter section headers are white */
    div[data-testid="stSidebar"] strong {
        color: white !important;
    }
    /* ── END SIDEBAR ──────────────────────────────────────────────────────── */
    
    /* ── MAIN CONTENT AREA — ALL TEXT DARK & VISIBLE ─────────────────────── */
    
    /* Force all main area text to be dark/black for contrast */
    .main {
        background: #F0F8FF !important;
    }
    
    .main * {
        color: #1a1a2e !important;
    }
    
    /* Input labels - make them clearly visible */
    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label,
    .stSlider label,
    .stMultiSelect label {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #0D47A1 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    
    /* Make all headings visible */
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main h5,
    .main h6 {
        color: #0D47A1 !important;
        font-weight: 800 !important;
    }
    
    /* Paragraph text visibility */
    .main p,
    .main div,
    .main span {
        color: #1a1a2e !important;
    }
    
    /* Button text */
    .main button,
    .stButton > button {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Metric and info boxes */
    .stMetric label {
        color: #0D47A1 !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #0D47A1 !important;
        font-weight: 800 !important;
    }
    
    /* Success, Error, Warning messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        color: #1a1a2e !important;
    }
    
    /* Tab labels - make them dark & visible */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0D47A1 !important;
        font-weight: 700 !important;
    }
    
    /* Dataframe text */
    .stDataFrame {
        font-size: 0.85rem !important;
        color: #1a1a2e !important;
    }
    
    /* ── END MAIN CONTENT AREA ─────────────────────────────────────────── */
    
    /* Param cards */
    .param-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .param-card .p-label {
        font-size: 0.75rem;
        color: #555 !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .param-card .p-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .param-card .p-unit {
        font-size: 0.75rem;
        color: #777 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }

    /* Bulk scanner */
    .bulk-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0D47A1 !important;
        margin-bottom: 0.8rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .upload-box {
        border: 2.5px dashed #4ECDC4;
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        background: #F0FFFE;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0D47A1 !important;
        font-weight: 700 !important;
    }
    
    /* General text visibility */
    .stMarkdown p {
        color: #1a1a2e !important;
        font-size: 1rem;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #666 !important;
        font-size: 0.85rem;
        padding: 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────────
def wqi_category(score):
    if score < 25:    return 'Excellent'
    elif score < 50:  return 'Good'
    elif score < 75:  return 'Poor'
    elif score < 100: return 'Very Poor'
    else:             return 'Unfit for Drinking'

def wqi_color(cat):
    return {'Excellent':'#1B5E20','Good':'#4CAF50','Poor':'#FF9800',
            'Very Poor':'#F44336','Unfit for Drinking':'#880E4F'}.get(cat, '#1565C0')

def get_status_class(category):
    return {
        'Excellent': 'status-excellent',
        'Good': 'status-good',
        'Poor': 'status-poor',
        'Very Poor': 'status-very-poor',
        'Unfit for Drinking': 'status-unfit'
    }.get(category, 'status-good')

CAT_COLOR_MAP = {'Excellent':'#1B5E20','Good':'#4CAF50','Poor':'#FF9800',
                 'Very Poor':'#F44336','Unfit for Drinking':'#880E4F'}

FEATURES = [
    'temp','do','ph','conductivity','bod','nitrate',
    'fecal_coliform','total_coliform',
    'do_ph_ratio','bod_do_ratio','coliform_log','fecal_log',
    'ph_deviation','do_deficit','conductivity_log'
]

def add_engineered_features(df):
    df = df.copy()
    df['do_ph_ratio']      = df['do'] / (df['ph'] + 1e-6)
    df['bod_do_ratio']     = df['bod'] / (df['do'] + 1e-6)
    df['coliform_log']     = np.log1p(df['total_coliform'])
    df['fecal_log']        = np.log1p(df['fecal_coliform'])
    df['ph_deviation']     = (df['ph'] - 7.0).abs()
    df['do_deficit']       = (14.6 - df['do']).clip(lower=0)
    df['conductivity_log'] = np.log1p(df['conductivity'])
    return df

def engineer_single(row_dict):
    d = row_dict.copy()
    d['do_ph_ratio']      = d['do'] / (d['ph'] + 1e-6)
    d['bod_do_ratio']     = d['bod'] / (d['do'] + 1e-6)
    d['coliform_log']     = np.log1p(d['total_coliform'])
    d['fecal_log']        = np.log1p(d['fecal_coliform'])
    d['ph_deviation']     = abs(d['ph'] - 7.0)
    d['do_deficit']       = max(0, 14.6 - d['do'])
    d['conductivity_log'] = np.log1p(d['conductivity'])
    return [d[f] for f in FEATURES]

def create_sample_csv():
    sample_data = {
        'location': ['Sample Site 1', 'Sample Site 2', 'Sample Site 3'],
        'state': ['Gujarat', 'Maharashtra', 'Karnataka'],
        'temp': [25.5, 28.2, 26.8],
        'do': [7.2, 5.8, 8.1],
        'ph': [7.1, 7.5, 6.9],
        'conductivity': [280, 450, 320],
        'bod': [1.5, 3.2, 2.1],
        'nitrate': [12.5, 18.3, 9.7],
        'fecal_coliform': [50, 150, 30],
        'total_coliform': [200, 500, 150]
    }
    df = pd.DataFrame(sample_data)
    return df.to_csv(index=False).encode('utf-8')

def validate_bulk_data(df):
    required_cols = ['temp','do','ph','conductivity','bod','nitrate','fecal_coliform','total_coliform']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    for col in required_cols:
        try:
            pd.to_numeric(df[col], errors='coerce')
        except:
            return False, f"Column '{col}' contains non-numeric values"
    return True, "Validation successful"

def predict_batch_wqi(df, reg_model, clf_model, scaler, label_enc):
    results = []
    for idx, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            feat_scaled = scaler.transform([engineer_single(row_dict)])
            wqi_pred = float(reg_model.predict(feat_scaled)[0])
            cat_pred = label_enc.inverse_transform(clf_model.predict(feat_scaled))[0]
            cat_prob = float(clf_model.predict_proba(feat_scaled)[0].max())
            results.append({
                'Location': row.get('location', f'Row {idx+1}'),
                'State': row.get('state', 'N/A'),
                'Temperature': f"{row['temp']:.1f}°C",
                'DO': f"{row['do']:.1f}",
                'pH': f"{row['ph']:.1f}",
                'Conductivity': f"{row['conductivity']:.0f}",
                'BOD': f"{row['bod']:.1f}",
                'Nitrate': f"{row['nitrate']:.1f}",
                'WQI_Score': round(wqi_pred, 2),
                'Category': cat_pred,
                'Confidence': f"{cat_prob*100:.1f}%"
            })
        except Exception as e:
            results.append({
                'Location': row.get('location', f'Row {idx+1}'),
                'State': row.get('state', 'N/A'),
                'WQI_Score': 'Error',
                'Category': 'Processing Error',
                'Confidence': '0%'
            })
    return pd.DataFrame(results)


# ── Load & Process Data ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('water_dataX.csv', encoding='latin1')
    df.columns = ['station_code','location','state','temp','do','ph',
                  'conductivity','bod','nitrate','fecal_coliform','total_coliform','year']
    num_cols = ['temp','do','ph','conductivity','bod','nitrate','fecal_coliform','total_coliform']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    imp = KNNImputer(n_neighbors=5)
    df[num_cols] = imp.fit_transform(df[num_cols])
    standards = {
        'do':{'ideal':14.6,'permissible':5.0,'weight':0.31},
        'ph':{'ideal':7.0,'permissible':8.5,'weight':0.22},
        'bod':{'ideal':0.0,'permissible':3.0,'weight':0.19},
        'nitrate':{'ideal':0.0,'permissible':45.0,'weight':0.10},
        'conductivity':{'ideal':0.0,'permissible':500.0,'weight':0.08},
        'temp':{'ideal':25.0,'permissible':30.0,'weight':0.10},
    }
    wqi = np.zeros(len(df))
    ws  = sum(v['weight'] for v in standards.values())
    for p, v in standards.items():
        qi = ((df[p] - v['ideal']) / (v['permissible'] - v['ideal'])) * 100
        wqi += v['weight'] * qi.clip(lower=0)
    df['WQI'] = (wqi / ws).round(2)
    df['WQI_Category'] = df['WQI'].apply(wqi_category)
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df = df[(df[col] >= Q1) & (df[col] <= Q3)]
    return df.reset_index(drop=True)


# ── Train Models Automatically ─────────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    df_ml  = add_engineered_features(_df)
    X      = df_ml[FEATURES]
    y      = df_ml['WQI']
    sc     = StandardScaler()
    X_sc   = sc.fit_transform(X)
    reg    = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                               max_depth=6, random_state=42, verbosity=0)
    reg.fit(X_sc, y)
    le     = LabelEncoder()
    y_cat  = le.fit_transform(df_ml['WQI_Category'])
    clf    = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_sc, y_cat)
    return reg, clf, sc, le


# ── Load Everything ────────────────────────────────────────────────────────────
with st.spinner('💧 Loading AquaVision... Training models, please wait ~30 seconds...'):
    df = load_data()
    reg_model, clf_model, scaler, label_enc = train_models(df)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 AquaVision")
    st.markdown("*Water Quality Intelligence*")
    st.divider()
    page = st.selectbox("📍 Navigate to", [
        "🏠 Dashboard",
        "🔬 Exploratory Analysis",
        "📈 Trends & Patterns",
        "🤖 WQI Predictor",
        "📊 Bulk Scanner",
        "🗺️ State Intelligence",
        "📋 Dataset Explorer",
    ])
    st.divider()
    st.markdown("**Filter Data**")
    years = sorted(df['year'].unique())
    year_range = st.select_slider("Year Range", options=years, value=(min(years), max(years)))
    states_avail = ['All'] + sorted(df['state'].dropna().unique().tolist())
    sel_state = st.selectbox("State", states_avail)
    st.divider()
    st.markdown("**ℹ️ WQI Scale**")
    for cat, col in [('< 25 — Excellent','#90EE90'),('25–50 — Good','#98FB98'),
                      ('50–75 — Poor','#FFD700'),('75–100 — Very Poor','#FF6347'),
                      ('> 100 — Unfit','#FF69B4')]:
        st.markdown(f"<span style='color:{col};font-size:1.2rem'>■</span> <span style='color:white;font-weight:600'>{cat}</span>", unsafe_allow_html=True)

mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
if sel_state != 'All':
    mask &= (df['state'] == sel_state)
dff = df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <h1>💧 AquaVision</h1>
        <p>Water Quality Index Analysis &amp; Predictive Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Records",    f"{len(dff):,}",                      "observations"),
        ("Avg WQI Score",    f"{dff['WQI'].mean():.1f}",           "overall quality"),
        ("States Covered",   f"{dff['state'].nunique()}",           "unique states"),
        ("Monitoring Sites", f"{dff['station_code'].nunique():,}", "stations"),
        ("Years Spanned",    f"{dff['year'].nunique()}",            "data years"),
    ]
    for col, (label, value, sub) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 WQI Category Breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cat_cnt = dff['WQI_Category'].value_counts().reset_index()
        cat_cnt.columns = ['Category', 'Count']
        fig = px.pie(cat_cnt, names='Category', values='Count',
                     color='Category', color_discrete_map=CAT_COLOR_MAP,
                     title='Distribution of WQI Categories', hole=0.42)
        fig.update_traces(textinfo='label+percent', pull=[0.04]*len(cat_cnt))
        fig.update_layout(template='plotly_white', showlegend=True, height=380,
                          font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", size=13))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        yr_wqi = dff.groupby('year')['WQI'].agg(['mean','std']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yr_wqi['year'], y=yr_wqi['mean']+yr_wqi['std'],
                                 fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=yr_wqi['year'], y=yr_wqi['mean']-yr_wqi['std'],
                                 fill='tonexty', mode='lines', line=dict(width=0),
                                 fillcolor='rgba(21,101,192,0.15)', name='±1 Std Dev'))
        fig.add_trace(go.Scatter(x=yr_wqi['year'], y=yr_wqi['mean'], mode='lines+markers',
                                 line=dict(color='#1565C0', width=3), marker=dict(size=7), name='Mean WQI'))
        fig.update_layout(title='Year-wise WQI Trend', xaxis_title='Year',
                          yaxis_title='WQI Score', template='plotly_white', height=380,
                          font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", size=13))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🧪 Parameter Overview</div>', unsafe_allow_html=True)

    param_info = {
        'temp': ('🌡️ Temperature', '°C', '#FF6B6B'),
        'do': ('💨 Dissolved Oxygen', 'mg/L', '#4ECDC4'),
        'ph': ('⚗️ pH Level', 'pH', '#45B7D1'),
        'conductivity': ('⚡ Conductivity', 'µS/cm', '#FFA07A'),
        'bod': ('🦠 BOD', 'mg/L', '#98D8C8'),
        'nitrate': ('🧪 Nitrate', 'mg/L', '#F7DC6F')
    }

    cols = st.columns(6)
    for i, (col_name, (label, unit, color)) in enumerate(param_info.items()):
        value = dff[col_name].mean()
        with cols[i]:
            st.markdown(f"""
            <div class="param-card" style="border-left: 5px solid {color};">
                <div class="p-label">{label}</div>
                <div class="p-value" style="color: {color};">{value:.2f}</div>
                <div class="p-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Exploratory Analysis":
    st.markdown('<div class="section-title">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔥 Correlations", "📦 Boxplots", "🔵 PCA"])
    num_cols_list = ['temp','do','ph','conductivity','bod','nitrate','fecal_coliform','total_coliform']
    font_cfg = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", size=13)

    with tab1:
        selected_param = st.selectbox("Select Parameter", num_cols_list)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f'{selected_param.upper()} Distribution',
                                            f'{selected_param.upper()} by WQI Category'])
        fig.add_trace(go.Histogram(x=dff[selected_param], nbinsx=50,
                                   marker_color='#1565C0', opacity=0.75, name=selected_param), row=1, col=1)
        for cat, color in CAT_COLOR_MAP.items():
            sub = dff[dff['WQI_Category']==cat][selected_param]
            if len(sub) > 0:
                fig.add_trace(go.Box(y=sub, name=cat, marker_color=color), row=1, col=2)
        fig.update_layout(template='plotly_white', height=420, showlegend=False, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr_matrix = dff[num_cols_list + ['WQI']].corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title='Feature Correlation Heatmap')
        fig.update_layout(template='plotly_white', height=520, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_subplots(rows=2, cols=4, subplot_titles=[c.upper() for c in num_cols_list])
        for idx, col_name in enumerate(num_cols_list):
            r, c = divmod(idx, 4)
            for cat, color in CAT_COLOR_MAP.items():
                sub = dff[dff['WQI_Category']==cat][col_name]
                if len(sub) > 0:
                    fig.add_trace(go.Box(y=sub, name=cat, marker_color=color,
                                         showlegend=(idx==0)), row=r+1, col=c+1)
        fig.update_layout(template='plotly_white', height=580,
                          title_text='Parameter Distribution by WQI Category', font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        sc_pca = StandardScaler()
        X_pca  = sc_pca.fit_transform(dff[num_cols_list].fillna(dff[num_cols_list].mean()))
        pca    = PCA(n_components=3)
        coords = pca.fit_transform(X_pca)
        pca_df = pd.DataFrame({'PC1':coords[:,0],'PC2':coords[:,1],'PC3':coords[:,2],
                               'Category':dff['WQI_Category'].values,'WQI':dff['WQI'].values})
        col_pca1, col_pca2 = st.columns([2,1])
        with col_pca1:
            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                                color='Category', color_discrete_map=CAT_COLOR_MAP,
                                opacity=0.65, size_max=6,
                                title=f'PCA 3D — {sum(pca.explained_variance_ratio_)*100:.1f}% Variance Explained')
            fig.update_layout(height=520, font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)
        with col_pca2:
            ev = pca.explained_variance_ratio_ * 100
            fig = go.Figure(go.Bar(x=[f'PC{i+1}' for i in range(3)], y=ev,
                                   marker_color=['#0D47A1','#1565C0','#42A5F5'],
                                   text=[f'{v:.1f}%' for v in ev], textposition='outside'))
            fig.update_layout(title='Explained Variance', template='plotly_white', height=300, font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends & Patterns":
    st.markdown('<div class="section-title">📈 Trends & Pattern Analysis</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["⏳ Time Trends", "🌊 Parameter Deep-Dive"])
    font_cfg = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", size=13)

    with tab1:
        params_sel = st.multiselect("Select parameters to plot",
                                    ['temp','do','ph','conductivity','bod','nitrate','WQI'],
                                    default=['WQI','do','ph'])
        yr_agg = dff.groupby('year')[params_sel].mean().reset_index()
        fig = go.Figure()
        for i, param in enumerate(params_sel):
            fig.add_trace(go.Scatter(x=yr_agg['year'], y=yr_agg[param],
                                     mode='lines+markers', name=param.upper(),
                                     line=dict(width=2.5, color=px.colors.qualitative.Bold[i % 10]),
                                     marker=dict(size=7)))
        fig.update_layout(title='Year-wise Multi-Parameter Trends',
                          xaxis_title='Year', template='plotly_white', height=450, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)

        cat_yr = dff.groupby(['year','WQI_Category']).size().reset_index(name='count')
        fig2 = px.area(cat_yr, x='year', y='count', color='WQI_Category',
                       color_discrete_map=CAT_COLOR_MAP, title='WQI Category Composition Over Years')
        fig2.update_layout(template='plotly_white', height=380, font=font_cfg)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        param_dd = st.selectbox("Parameter", ['temp','do','ph','conductivity','bod','nitrate'])
        top_n    = st.slider("Top N States", 5, 20, 10)
        state_param = dff.groupby('state')[param_dd].mean().sort_values(ascending=False).head(top_n)
        fig = px.bar(state_param.reset_index(), x='state', y=param_dd,
                     title=f'Top {top_n} States by avg {param_dd.upper()}',
                     color=param_dd, color_continuous_scale='Blues', text_auto='.2f')
        fig.update_layout(template='plotly_white', height=400, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BULK SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Bulk Scanner":
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Bulk WQI Scanner</h1>
        <p>Batch Process &amp; Predict Water Quality Index for Multiple Samples</p>
    </div>
    """, unsafe_allow_html=True)

    if 'bulk_results' not in st.session_state:
        st.session_state.bulk_results = None
    if 'scan_count' not in st.session_state:
        st.session_state.scan_count = 0

    col_download, col_spacer, col_upload, col_spacer2, col_download_results = st.columns([1.2, 0.3, 1.2, 0.3, 1.2])

    with col_download:
        st.markdown('<div class="bulk-header">📥 Step 1: Download Sample File</div>', unsafe_allow_html=True)
        st.write("Download this template to see the required format for your data.")
        if st.download_button(
            label="📄 Download Sample CSV",
            data=create_sample_csv(),
            file_name="aquavision_sample.csv",
            mime="text/csv",
            use_container_width=True,
            key="sample_download"
        ):
            st.success("✅ Sample file ready!")

    with col_upload:
        st.markdown('<div class="bulk-header">📤 Step 2: Upload Your Data</div>', unsafe_allow_html=True)
        st.write("Upload a CSV or Excel file with the required water quality columns.")
        uploaded_file = st.file_uploader(
            "Drag and drop or browse (CSV, XLSX, XLS — max 200MB)",
            type=['csv', 'xlsx', 'xls'],
            key='bulk_upload',
        )

    with col_download_results:
        st.markdown('<div class="bulk-header">📊 Step 3: Download Results</div>', unsafe_allow_html=True)
        st.write("After scanning, download the full predictions as a CSV file.")
        if st.session_state.bulk_results is not None:
            csv_data = st.session_state.bulk_results.to_csv(index=False).encode('utf-8')
            if st.download_button(
                label="📥 Download Scanned Results",
                data=csv_data,
                file_name="aquavision_bulk_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="results_download"
            ):
                st.success("✅ Results downloaded!")
        else:
            st.button("📥 Download Results (scan first)", disabled=True, use_container_width=True)

    st.divider()

    if uploaded_file is not None:
        st.markdown('<div class="section-title">📋 File Preview & Validation</div>', unsafe_allow_html=True)
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            col_info1, col_info2, col_info3 = st.columns(3)
            col_info1.metric("📊 Total Records", len(df_upload))
            col_info2.metric("📈 Total Columns", len(df_upload.columns))
            col_info3.metric("✓ File Size", f"{uploaded_file.size / 1024:.1f} KB")

            is_valid, validation_msg = validate_bulk_data(df_upload)

            if is_valid:
                st.success(f"✅ {validation_msg}")
                st.markdown("**Preview — First 5 Rows:**")
                st.dataframe(df_upload.head(), use_container_width=True)

                col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
                with col_btn2:
                    scan_button = st.button("🚀 SCAN & PREDICT WQI", type="primary",
                                            use_container_width=True, key="scan_btn")

                if scan_button:
                    with st.spinner('⏳ Processing your data...'):
                        st.session_state.bulk_results = predict_batch_wqi(
                            df_upload, reg_model, clf_model, scaler, label_enc
                        )
                        st.session_state.scan_count = len(st.session_state.bulk_results)
                    st.success(f"✅ Scan completed! Processed {st.session_state.scan_count} records.")
            else:
                st.error(f"❌ Validation Error: {validation_msg}")
                st.info("Required columns: temp, do, ph, conductivity, bod, nitrate, fecal_coliform, total_coliform")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    if st.session_state.bulk_results is not None:
        st.markdown('<div class="section-title">📊 Scan Results</div>', unsafe_allow_html=True)
        results_df = st.session_state.bulk_results.copy()
        font_cfg = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", size=13)

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        avg_wqi = pd.to_numeric(results_df['WQI_Score'], errors='coerce').mean()
        excellent_count = len(results_df[results_df['Category'] == 'Excellent'])
        poor_count = len(results_df[results_df['Category'].isin(['Poor', 'Very Poor', 'Unfit for Drinking'])])
        success_rate = len(results_df[results_df['Category'] != 'Processing Error']) / len(results_df) * 100

        col_stat1.metric("📊 Average WQI", f"{avg_wqi:.2f}")
        col_stat2.metric("💚 Excellent Quality", excellent_count)
        col_stat3.metric("⚠️ Needs Attention", poor_count)
        col_stat4.metric("✅ Success Rate", f"{success_rate:.1f}%")

        st.divider()
        st.markdown("**Detailed Results Table:**")
        st.dataframe(results_df, use_container_width=True, height=400,
                     column_config={'WQI_Score': st.column_config.NumberColumn(format="%.2f")})
        st.divider()

        st.markdown('<div class="section-title">📈 Results Visualization</div>', unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            cat_dist = results_df['Category'].value_counts().reset_index()
            cat_dist.columns = ['Category', 'Count']
            cat_dist = cat_dist[cat_dist['Category'] != 'Processing Error']
            fig = px.bar(cat_dist, x='Category', y='Count', color='Category',
                         color_discrete_map=CAT_COLOR_MAP, title='WQI Category Distribution', text_auto=True)
            fig.update_layout(template='plotly_white', height=380, showlegend=False, font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            wqi_scores = pd.to_numeric(results_df['WQI_Score'], errors='coerce').dropna()
            fig = px.histogram(x=wqi_scores, nbins=30, title='WQI Score Distribution',
                               labels={'x': 'WQI Score', 'count': 'Frequency'})
            fig.update_traces(marker_color='#1565C0')
            fig.update_layout(template='plotly_white', height=380, font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)

        if 'State' in results_df.columns and results_df['State'].notna().sum() > 0:
            st.markdown('<div class="section-title">🗺️ State-wise Summary</div>', unsafe_allow_html=True)
            state_summary = results_df.groupby('State').agg(
                {'WQI_Score': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                 'Category': 'count'}
            ).round(2)
            state_summary.columns = ['Avg_WQI', 'Sample_Count']
            state_summary = state_summary.sort_values('Avg_WQI')
            fig = px.bar(state_summary.reset_index(), x='State', y='Avg_WQI',
                         color='Avg_WQI', color_continuous_scale='RdYlGn_r',
                         title='Average WQI by State', text_auto='.2f')
            fig.update_layout(template='plotly_white', height=350, font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WQI PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 WQI Predictor":
    st.markdown('<div class="section-title">🤖 AI-Powered WQI Predictor</div>', unsafe_allow_html=True)
    st.write("Enter water parameter values below to predict the Water Quality Index:")

    st.markdown('<div class="predict-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        temp    = st.number_input("🌡️ Temperature (°C)",           0.0, 50.0,     25.0, 0.1)
        do      = st.number_input("💨 Dissolved Oxygen (mg/L)",    0.0, 20.0,      7.5, 0.1)
        ph      = st.number_input("⚗️ pH Level",                   0.0, 14.0,      7.0, 0.1)
    with col2:
        cond    = st.number_input("⚡ Conductivity (µmhos/cm)",    0.0, 5000.0,  250.0, 1.0)
        bod     = st.number_input("🦠 BOD (mg/L)",                 0.0, 100.0,     2.0, 0.1)
        nitrate = st.number_input("🧪 Nitrate (mg/L)",             0.0, 200.0,     5.0, 0.1)
    with col3:
        fecal   = st.number_input("🔬 Fecal Coliform (MPN/100mL)", 0.0, 200000.0,100.0, 1.0)
        total   = st.number_input("🔭 Total Coliform (MPN/100mL)", 0.0, 200000.0,500.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Predict Water Quality Index", use_container_width=True, type="primary"):
        font_cfg = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", size=13)
        row = {'temp':temp,'do':do,'ph':ph,'conductivity':cond,
               'bod':bod,'nitrate':nitrate,'fecal_coliform':fecal,'total_coliform':total}
        feat_scaled = scaler.transform([engineer_single(row)])
        wqi_pred    = float(reg_model.predict(feat_scaled)[0])
        cat_pred    = label_enc.inverse_transform(clf_model.predict(feat_scaled))[0]
        cat_prob    = clf_model.predict_proba(feat_scaled)[0]
        cat_color   = wqi_color(cat_pred)

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{cat_color}">
                <div class="label">Predicted WQI Score</div>
                <div class="value" style="color:{cat_color}">{wqi_pred:.2f}</div>
                <div class="sub">Water Quality Index</div>
                <span class="wqi-badge" style="background:{cat_color}">{cat_pred}</span>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=min(wqi_pred, 120),
                domain={'x':[0,1],'y':[0,1]}, title={'text':"WQI Gauge"},
                gauge={'axis':{'range':[0,120]},
                       'bar':{'color':cat_color},
                       'steps':[{'range':[0,25],'color':'#E8F5E9'},
                                 {'range':[25,50],'color':'#F1F8E9'},
                                 {'range':[50,75],'color':'#FFF3E0'},
                                 {'range':[75,100],'color':'#FFEBEE'},
                                 {'range':[100,120],'color':'#FCE4EC'}],
                       'threshold':{'line':{'color':'black','width':3},
                                    'thickness':0.8,'value':wqi_pred}}))
            fig.update_layout(height=250, margin=dict(t=30,b=10,l=20,r=20), font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)
        with col_r3:
            probs = pd.DataFrame({'Category':label_enc.classes_,
                                  'Probability':cat_prob*100}).sort_values('Probability', ascending=True)
            fig = px.bar(probs, x='Probability', y='Category', orientation='h',
                         title='Category Confidence (%)', color='Probability',
                         color_continuous_scale='Blues', text_auto='.1f')
            fig.update_layout(height=250, template='plotly_white',
                              margin=dict(t=40,b=10), showlegend=False, font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**📊 Your Sample vs Dataset Average**")
        avg = dff[['temp','do','ph','conductivity','bod','nitrate']].mean()
        compare = pd.DataFrame({
            'Parameter': list(avg.index),
            'Your Value': [temp, do, ph, cond, bod, nitrate],
            'Dataset Average': avg.values
        }).melt(id_vars='Parameter', var_name='Source', value_name='Value')
        fig = px.bar(compare, x='Parameter', y='Value', color='Source', barmode='group',
                     template='plotly_white', color_discrete_sequence=['#1565C0','#80DEEA'],
                     title='Input vs Dataset Average Comparison')
        fig.update_layout(height=340, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ State Intelligence":
    st.markdown('<div class="section-title">🗺️ State-wise Water Quality Intelligence</div>', unsafe_allow_html=True)
    font_cfg = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", size=13)

    state_stats = dff.groupby('state').agg(
        avg_wqi=('WQI','mean'), min_wqi=('WQI','min'), max_wqi=('WQI','max'),
        records=('WQI','count'), avg_do=('do','mean'), avg_ph=('ph','mean'), avg_bod=('bod','mean'),
    ).reset_index().round(2)
    state_stats['category'] = state_stats['avg_wqi'].apply(wqi_category)

    col1, col2 = st.columns([2,1])
    with col1:
        fig = px.bar(state_stats.sort_values('avg_wqi', ascending=False),
                     x='state', y='avg_wqi', color='category',
                     color_discrete_map=CAT_COLOR_MAP, title='Average WQI by State',
                     text='avg_wqi', labels={'avg_wqi':'Average WQI','state':'State'})
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(template='plotly_white', height=450,
                          xaxis={'categoryorder':'total descending'}, xaxis_tickangle=-35, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.treemap(state_stats, path=['state'], values='records',
                          color='avg_wqi', color_continuous_scale='RdYlGn_r',
                          title='State Coverage', hover_data=['avg_wqi','category'])
        fig2.update_layout(height=450, font=font_cfg)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**📋 State Summary Table**")
    st.dataframe(state_stats.sort_values('avg_wqi', ascending=False).reset_index(drop=True),
                 use_container_width=True, height=350)

    top5 = state_stats.nlargest(5, 'records')['state'].tolist()
    radar_params = ['avg_do','avg_ph','avg_bod','avg_wqi']
    fig_radar = go.Figure()
    for i, st_name in enumerate(top5):
        row = state_stats[state_stats['state']==st_name].iloc[0]
        vals = [float(row[p]) for p in radar_params] + [float(row[radar_params[0]])]
        fig_radar.add_trace(go.Scatterpolar(r=vals, theta=radar_params+[radar_params[0]],
                                             fill='toself', name=st_name,
                                             line=dict(color=['#1565C0','#00897B','#E53935','#7B1FA2','#F57C00'][i]),
                                             opacity=0.6))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
                             title='Top 5 States — Parameter Radar',
                             template='plotly_white', height=420, font=font_cfg)
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Explorer":
    st.markdown('<div class="section-title">📋 Dataset Explorer</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    with col1:
        search = st.text_input("🔍 Search by location or state")
    with col2:
        cat_filter = st.selectbox("Filter by WQI Category", ['All'] + list(dff['WQI_Category'].unique()))

    df_show = dff.copy()
    if search:
        df_show = df_show[df_show['location'].str.contains(search, case=False, na=False) |
                          df_show['state'].str.contains(search, case=False, na=False)]
    if cat_filter != 'All':
        df_show = df_show[df_show['WQI_Category'] == cat_filter]

    st.markdown(f"Showing **{len(df_show):,}** records")
    st.dataframe(df_show[['station_code','location','state','year','temp','do','ph',
                           'conductivity','bod','nitrate','WQI','WQI_Category']].reset_index(drop=True),
                 use_container_width=True, height=480)
    st.download_button(label="⬇️ Download Filtered Data as CSV",
                       data=df_show.to_csv(index=False).encode('utf-8'),
                       file_name='aquavision_filtered.csv', mime='text/csv')


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer-text">
    💧 <strong>AquaVision</strong> — Water Quality Intelligence Platform &nbsp;|&nbsp;
    Final Year Project &nbsp;|&nbsp; Built with Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
