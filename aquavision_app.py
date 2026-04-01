"""
💧 AquaVision — Water Quality Intelligence
Final Year Project | Professional Streamlit Dashboard App
Run: streamlit run aquavision_professional.py
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

# ── Professional Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Professional Color Palette */
    :root {
        --primary: #1e3a8a;        /* Deep Blue */
        --secondary: #0f766e;      /* Teal */
        --accent: #d97706;         /* Amber */
        --success: #059669;        /* Green */
        --warning: #f97316;        /* Orange */
        --danger: #dc2626;         /* Red */
        --light-bg: #f8fafc;       /* Very Light Gray */
        --dark-text: #0f172a;      /* Dark Slate */
    }
    
    /* Use system fonts for guaranteed rendering */
    html, body, [class*="css"], .stMarkdown, .stText, p, div, span, h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif !important;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    }
    
    /* Professional Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 50%, #0c4a3c 100%);
        padding: 3rem 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.25);
        border-top: 4px solid #d97706;
    }
    .hero-banner h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: white !important;
        margin: 0;
        letter-spacing: 0px;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .hero-banner p {
        font-size: 1rem;
        color: rgba(255,255,255,0.85) !important;
        margin: 0.8rem 0 0;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
        font-weight: 400;
    }

    /* Metric Cards - Professional Look */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1e3a8a;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #64748b !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a !important;
        margin: 0.5rem 0;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #94a3b8 !important;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    
    /* Status Badges */
    .wqi-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        color: white !important;
        margin-top: 0.5rem;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Predict Box */
    .predict-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #ecfdf5 100%);
        border-radius: 12px;
        padding: 2.5rem;
        border: 2px solid #bfdbfe;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Section Titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e3a8a !important;
        margin: 2rem 0 1.2rem;
        padding: 0.8rem 1rem;
        border-left: 4px solid #d97706;
        background: #fef3c7;
        border-radius: 0 6px 6px 0;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }

    /* ══════════════════════════════════════════════════════════════════ */
    /* SIDEBAR - Professional Dark Theme */
    /* ══════════════════════════════════════════════════════════════════ */
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e293b 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* All sidebar text - bright white */
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
        font-weight: 500 !important;
    }
    
    /* Sidebar Labels - Bold */
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stNumberInput label,
    div[data-testid="stSidebar"] .stTextInput label {
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: capitalize;
    }
    
    /* Selectbox Styling */
    div[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.1) !important;
        border: 1.5px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 6px !important;
    }
    
    div[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: white !important;
    }
    
    /* Slider */
    div[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    div[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: rgba(255,255,255,0.7) !important;
    }
    
    /* Divider */
    div[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Markdown in Sidebar */
    div[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }
    
    div[data-testid="stSidebar"] strong {
        color: #fbbf24 !important;
        font-weight: 700 !important;
    }
    
    /* ══════════════════════════════════════════════════════════════════ */
    /* MAIN CONTENT AREA */
    /* ══════════════════════════════════════════════════════════════════ */
    
    /* Input Labels */
    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label,
    .stSlider label,
    .stMultiSelect label {
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        color: #1e3a8a !important;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    
    /* Headings */
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main h5,
    .main h6 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    
    /* Paragraph Text */
    .main p,
    .stMarkdown p {
        color: #334155 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Download/Primary Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .stMetric label {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #1e3a8a !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    
    /* Messages */
    .stSuccess {
        background-color: #ecfdf5 !important;
        border-left: 4px solid #059669 !important;
        color: #065f46 !important;
    }
    
    .stError {
        background-color: #fef2f2 !important;
        border-left: 4px solid #dc2626 !important;
        color: #7f1d1d !important;
    }
    
    .stWarning {
        background-color: #fffbeb !important;
        border-left: 4px solid #d97706 !important;
        color: #78350f !important;
    }
    
    .stInfo {
        background-color: #f0f9ff !important;
        border-left: 4px solid #0284c7 !important;
        color: #0c2340 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1e3a8a !important;
        font-weight: 700 !important;
        border-bottom-color: #d97706 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        font-size: 0.85rem !important;
        color: #334155 !important;
    }
    
    /* Param Cards */
    .param-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .param-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .param-card .p-label {
        font-size: 0.75rem;
        color: #64748b !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    
    .param-card .p-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    
    .param-card .p-unit {
        font-size: 0.75rem;
        color: #94a3b8 !important;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }

    /* Bulk Scanner Header */
    .bulk-header {
        font-size: 1rem;
        font-weight: 700;
        color: #1e3a8a !important;
        margin-bottom: 0.8rem;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    }
    
    /* Upload Box */
    .upload-box {
        border: 2px dashed #0f766e;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #d97706;
        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #94a3b8 !important;
        font-size: 0.85rem;
        padding: 1.5rem;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
        border-top: 1px solid #e2e8f0;
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
    colors = {
        'Excellent': '#059669',      # Green
        'Good': '#0891b2',           # Cyan
        'Poor': '#f97316',           # Orange
        'Very Poor': '#dc2626',      # Red
        'Unfit for Drinking': '#7c2d12'  # Dark Red
    }
    return colors.get(cat, '#1e3a8a')

CAT_COLOR_MAP = {
    'Excellent': '#059669',
    'Good': '#0891b2',
    'Poor': '#f97316',
    'Very Poor': '#dc2626',
    'Unfit for Drinking': '#7c2d12'
}

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
with st.spinner('⏳ Loading AquaVision... Training models...'):
    df = load_data()
    reg_model, clf_model, scaler, label_enc = train_models(df)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💧 AquaVision")
    st.markdown("*Water Quality Intelligence*")
    st.divider()
    page = st.selectbox("📍 Navigate", [
        "🏠 Dashboard",
        "🔬 Analysis",
        "📈 Trends",
        "🤖 Predictor",
        "📊 Bulk Scanner",
        "🗺️ State Intel",
        "📋 Explorer",
    ])
    st.divider()
    st.markdown("**Filter Data**")
    years = sorted(df['year'].unique())
    year_range = st.select_slider("Year Range", options=years, value=(min(years), max(years)))
    states_avail = ['All'] + sorted(df['state'].dropna().unique().tolist())
    sel_state = st.selectbox("State", states_avail)
    st.divider()
    st.markdown("**WQI Scale**")
    for cat, col in [('< 25 — Excellent','#059669'),('25–50 — Good','#0891b2'),
                      ('50–75 — Poor','#f97316'),('75–100 — Very Poor','#dc2626'),
                      ('> 100 — Unfit','#7c2d12')]:
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
        <p>Water Quality Index Analysis & Predictive Intelligence Platform</p>
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

    st.markdown('<div class="section-title">📊 WQI Category Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cat_cnt = dff['WQI_Category'].value_counts().reset_index()
        cat_cnt.columns = ['Category', 'Count']
        fig = px.pie(cat_cnt, names='Category', values='Count',
                     color='Category', color_discrete_map=CAT_COLOR_MAP,
                     title='Distribution Overview', hole=0.42)
        fig.update_traces(textinfo='label+percent', pull=[0.04]*len(cat_cnt))
        fig.update_layout(template='plotly_white', showlegend=True, height=380,
                          font=dict(family="'Segoe UI', Roboto, Arial", size=12))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        yr_wqi = dff.groupby('year')['WQI'].agg(['mean','std']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yr_wqi['year'], y=yr_wqi['mean']+yr_wqi['std'],
                                 fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=yr_wqi['year'], y=yr_wqi['mean']-yr_wqi['std'],
                                 fill='tonexty', mode='lines', line=dict(width=0),
                                 fillcolor='rgba(30, 58, 138, 0.15)', name='±1 Std Dev'))
        fig.add_trace(go.Scatter(x=yr_wqi['year'], y=yr_wqi['mean'], mode='lines+markers',
                                 line=dict(color='#1e3a8a', width=3), marker=dict(size=7), name='Mean WQI'))
        fig.update_layout(title='Year-wise Trend', xaxis_title='Year',
                          yaxis_title='WQI Score', template='plotly_white', height=380,
                          font=dict(family="'Segoe UI', Roboto, Arial", size=12))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🧪 Parameter Overview</div>', unsafe_allow_html=True)

    param_info = {
        'temp': ('🌡️ Temperature', '°C', '#059669'),
        'do': ('💨 Dissolved Oxygen', 'mg/L', '#0891b2'),
        'ph': ('⚗️ pH Level', 'pH', '#0f766e'),
        'conductivity': ('⚡ Conductivity', 'µS/cm', '#d97706'),
        'bod': ('🦠 BOD', 'mg/L', '#f97316'),
        'nitrate': ('🧪 Nitrate', 'mg/L', '#1e3a8a')
    }

    cols = st.columns(6)
    for i, (col_name, (label, unit, color)) in enumerate(param_info.items()):
        value = dff[col_name].mean()
        with cols[i]:
            st.markdown(f"""
            <div class="param-card" style="border-top: 3px solid {color};">
                <div class="p-label">{label}</div>
                <div class="p-value" style="color: {color};">{value:.2f}</div>
                <div class="p-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Analysis":
    st.markdown('<div class="section-title">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔥 Correlations", "📦 Boxplots"])
    num_cols_list = ['temp','do','ph','conductivity','bod','nitrate','fecal_coliform','total_coliform']
    font_cfg = dict(family="'Segoe UI', Roboto, Arial", size=12)

    with tab1:
        selected_param = st.selectbox("Select Parameter", num_cols_list, key="dist_param")
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f'{selected_param.upper()} Distribution',
                                            f'{selected_param.upper()} by Category'])
        fig.add_trace(go.Histogram(x=dff[selected_param], nbinsx=50,
                                   marker_color='#1e3a8a', opacity=0.75, name=selected_param), row=1, col=1)
        for cat, color in CAT_COLOR_MAP.items():
            sub = dff[dff['WQI_Category']==cat][selected_param]
            if len(sub) > 0:
                fig.add_trace(go.Box(y=sub, name=cat, marker_color=color), row=1, col=2)
        fig.update_layout(template='plotly_white', height=400, showlegend=False, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr_matrix = dff[num_cols_list + ['WQI']].corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title='Feature Correlation Matrix')
        fig.update_layout(template='plotly_white', height=500, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        selected_box = st.selectbox("Parameter", num_cols_list, key="box_param")
        fig = go.Figure()
        for cat, color in CAT_COLOR_MAP.items():
            sub = dff[dff['WQI_Category']==cat][selected_box]
            if len(sub) > 0:
                fig.add_trace(go.Box(y=sub, name=cat, marker_color=color))
        fig.update_layout(template='plotly_white', height=400, title=f'{selected_box.upper()} Distribution', font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.markdown('<div class="section-title">📈 Trends & Patterns</div>', unsafe_allow_html=True)
    font_cfg = dict(family="'Segoe UI', Roboto, Arial", size=12)

    params_sel = st.multiselect("Select parameters",
                                ['temp','do','ph','conductivity','bod','nitrate','WQI'],
                                default=['WQI','do','ph'], key="trend_params")
    if params_sel:
        yr_agg = dff.groupby('year')[params_sel].mean().reset_index()
        fig = go.Figure()
        for i, param in enumerate(params_sel):
            fig.add_trace(go.Scatter(x=yr_agg['year'], y=yr_agg[param],
                                     mode='lines+markers', name=param.upper(),
                                     line=dict(width=2.5),
                                     marker=dict(size=6)))
        fig.update_layout(title='Multi-Parameter Trends',
                          xaxis_title='Year', template='plotly_white', height=400, font=font_cfg)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BULK SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Bulk Scanner":
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Bulk WQI Scanner</h1>
        <p>Process Multiple Water Samples at Once</p>
    </div>
    """, unsafe_allow_html=True)

    if 'bulk_results' not in st.session_state:
        st.session_state.bulk_results = None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="bulk-header">📥 Download Template</div>', unsafe_allow_html=True)
        if st.download_button(
            label="📄 Sample CSV",
            data=create_sample_csv(),
            file_name="sample.csv",
            mime="text/csv",
            use_container_width=True
        ):
            st.success("✅ Ready!")

    with col2:
        st.markdown('<div class="bulk-header">📤 Upload File</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "CSV, Excel, or JSON",
            type=['csv', 'xlsx', 'xls', 'json'],
            key='bulk_upload',
        )

    with col3:
        st.markdown('<div class="bulk-header">📊 Download Results</div>', unsafe_allow_html=True)
        if st.session_state.bulk_results is not None:
            csv_data = st.session_state.bulk_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Results CSV",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📥 Results (scan first)", disabled=True, use_container_width=True)

    st.divider()

    if uploaded_file is not None:
        st.markdown('<div class="section-title">📋 File Validation</div>', unsafe_allow_html=True)
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df_upload = pd.read_json(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            col_info1, col_info2, col_info3 = st.columns(3)
            col_info1.metric("📊 Records", len(df_upload))
            col_info2.metric("📈 Columns", len(df_upload.columns))
            col_info3.metric("📦 Size", f"{uploaded_file.size / 1024:.1f} KB")

            is_valid, validation_msg = validate_bulk_data(df_upload)

            if is_valid:
                st.success(f"✅ {validation_msg}")
                st.dataframe(df_upload.head(), use_container_width=True)

                if st.button("🚀 SCAN", use_container_width=True, type="primary"):
                    with st.spinner('⏳ Processing...'):
                        st.session_state.bulk_results = predict_batch_wqi(
                            df_upload, reg_model, clf_model, scaler, label_enc
                        )
                    st.success(f"✅ Done! {len(st.session_state.bulk_results)} records processed")
            else:
                st.error(f"❌ {validation_msg}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    if st.session_state.bulk_results is not None:
        st.markdown('<div class="section-title">📊 Results</div>', unsafe_allow_html=True)
        results_df = st.session_state.bulk_results.copy()
        font_cfg = dict(family="'Segoe UI', Roboto, Arial", size=12)

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        avg_wqi = pd.to_numeric(results_df['WQI_Score'], errors='coerce').mean()
        excellent_count = len(results_df[results_df['Category'] == 'Excellent'])
        poor_count = len(results_df[results_df['Category'].isin(['Poor', 'Very Poor', 'Unfit for Drinking'])])
        success_rate = len(results_df[results_df['Category'] != 'Processing Error']) / len(results_df) * 100

        col_stat1.metric("📊 Avg WQI", f"{avg_wqi:.2f}")
        col_stat2.metric("✅ Excellent", excellent_count)
        col_stat3.metric("⚠️ Needs Attention", poor_count)
        col_stat4.metric("✓ Success", f"{success_rate:.0f}%")

        st.dataframe(results_df, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WQI PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictor":
    st.markdown('<div class="section-title">🤖 WQI Predictor</div>', unsafe_allow_html=True)

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

    if st.button("🚀 Predict", use_container_width=True, type="primary"):
        font_cfg = dict(family="'Segoe UI', Roboto, Arial", size=12)
        row = {'temp':temp,'do':do,'ph':ph,'conductivity':cond,
               'bod':bod,'nitrate':nitrate,'fecal_coliform':fecal,'total_coliform':total}
        feat_scaled = scaler.transform([engineer_single(row)])
        wqi_pred    = float(reg_model.predict(feat_scaled)[0])
        cat_pred    = label_enc.inverse_transform(clf_model.predict(feat_scaled))[0]
        cat_prob    = clf_model.predict_proba(feat_scaled)[0]
        cat_color   = wqi_color(cat_pred)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{cat_color}">
                <div class="label">Predicted WQI Score</div>
                <div class="value" style="color:{cat_color}">{wqi_pred:.2f}</div>
                <span class="wqi-badge" style="background:{cat_color}">{cat_pred}</span>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=min(wqi_pred, 120),
                domain={'x':[0,1],'y':[0,1]}, title={'text':"WQI"},
                gauge={'axis':{'range':[0,120]},
                       'bar':{'color':cat_color},
                       'steps':[{'range':[0,25],'color':'#dbeafe'},
                                 {'range':[25,50],'color':'#ecfdf5'},
                                 {'range':[50,75],'color':'#fef3c7'},
                                 {'range':[75,100],'color':'#fee2e2'},
                                 {'range':[100,120],'color':'#fce7f3'}]}))
            fig.update_layout(height=280, margin=dict(t=30,b=10), font=font_cfg)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ State Intel":
    st.markdown('<div class="section-title">🗺️ State Intelligence</div>', unsafe_allow_html=True)
    font_cfg = dict(family="'Segoe UI', Roboto, Arial", size=12)

    state_stats = dff.groupby('state').agg(
        avg_wqi=('WQI','mean'), records=('WQI','count'), avg_do=('do','mean'), avg_ph=('ph','mean')
    ).reset_index().round(2)
    state_stats['category'] = state_stats['avg_wqi'].apply(wqi_category)

    fig = px.bar(state_stats.sort_values('avg_wqi', ascending=False),
                 x='state', y='avg_wqi', color='category',
                 color_discrete_map=CAT_COLOR_MAP, title='Average WQI by State',
                 text='avg_wqi')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-35, font=font_cfg)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(state_stats.sort_values('avg_wqi', ascending=False), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Explorer":
    st.markdown('<div class="section-title">📋 Dataset Explorer</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    with col1:
        search = st.text_input("🔍 Search location or state")
    with col2:
        cat_filter = st.selectbox("Category", ['All'] + list(dff['WQI_Category'].unique()))

    df_show = dff.copy()
    if search:
        df_show = df_show[df_show['location'].str.contains(search, case=False, na=False) |
                          df_show['state'].str.contains(search, case=False, na=False)]
    if cat_filter != 'All':
        df_show = df_show[df_show['WQI_Category'] == cat_filter]

    st.markdown(f"**{len(df_show):,}** records found")
    st.dataframe(df_show[['station_code','location','state','year','temp','do','ph',
                           'conductivity','bod','nitrate','WQI','WQI_Category']].reset_index(drop=True),
                 use_container_width=True, height=400)
    st.download_button(label="⬇️ Download CSV",
                       data=df_show.to_csv(index=False).encode('utf-8'),
                       file_name='data.csv', mime='text/csv')


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer-text">
    💧 <strong>AquaVision</strong> — Water Quality Intelligence Platform | Professional Edition
</div>
""", unsafe_allow_html=True)
