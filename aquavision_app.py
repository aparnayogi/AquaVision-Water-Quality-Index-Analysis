"""
💧 AquaVision — Water Quality Intelligence
Simplified & Redesigned | Bulk Scanner + Predictor + Model Insight
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import xgboost as xgb

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaVision | Water Quality Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Pinkish Professional Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [class*="css"], .stMarkdown, .stText, p, div, span, h1, h2, h3 {
        font-family: 'Poppins', 'Segoe UI', sans-serif !important;
    }
    
    /* ── MAIN BACKGROUND ── */
    .main {
        background: linear-gradient(135deg, #fef5f8 0%, #fdf2f7 50%, #f8f0f5 100%) !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fef5f8 0%, #fdf2f7 50%, #f8f0f5 100%) !important;
    }
    
    /* ── HERO BANNER ── */
    .hero-banner {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 50%, #be185d 100%);
        padding: 3.5rem 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 0 0 3rem 0;
        box-shadow: 0 20px 60px rgba(236, 72, 153, 0.35);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: drift 20s linear infinite;
    }
    
    @keyframes drift {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    
    .hero-banner h1 {
        font-size: 3.2rem;
        font-weight: 800;
        color: white !important;
        margin: 0;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .hero-banner p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9) !important;
        margin: 1rem 0 0;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* ── CARD STYLES ── */
    .card {
        background: white;
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(236, 72, 153, 0.08);
        border: 1px solid rgba(236, 72, 153, 0.1);
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
        margin-bottom: 2rem;
    }
    
    .card:hover {
        box-shadow: 0 16px 48px rgba(236, 72, 153, 0.15);
        transform: translateY(-4px);
        border-color: rgba(236, 72, 153, 0.2);
    }
    
    /* ── METRIC CARD ── */
    .metric-card {
        background: linear-gradient(135deg, #fff5f7 0%, #ffe4ed 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(236, 72, 153, 0.15);
        text-align: center;
        margin-bottom: 1.2rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #be185d !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ec4899 !important;
        margin: 0.8rem 0;
    }
    
    .metric-sub {
        font-size: 0.9rem;
        color: #be185d !important;
        opacity: 0.7;
    }
    
    /* ── SECTION TITLE ── */
    .section-title {
        font-size: 2rem;
        font-weight: 800;
        color: #be185d !important;
        margin: 2.5rem 0 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #ec4899;
        display: inline-block;
    }
    
    /* ── BADGES ── */
    .wqi-badge {
        display: inline-block;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 1rem;
    }
    
    /* ── INPUT STYLING ── */
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox select {
        border-radius: 12px !important;
        border: 2px solid rgba(236, 72, 153, 0.2) !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput input:focus,
    .stTextInput input:focus,
    .stSelectbox select:focus {
        border-color: #ec4899 !important;
        box-shadow: 0 0 0 3px rgba(236, 72, 153, 0.1) !important;
    }
    
    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label,
    .stSlider label {
        color: #be185d !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    /* ── BUTTONS ── */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2.5rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(236, 72, 153, 0.3) !important;
    }
    
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(236, 72, 153, 0.4) !important;
    }
    
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* ── TABS ── */
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #be185d !important;
        border-radius: 12px 12px 0 0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ec4899 !important;
        border-bottom-color: #ec4899 !important;
    }
    
    /* ── DATA FRAME ── */
    .stDataFrame {
        font-size: 0.95rem !important;
        color: #334155 !important;
    }
    
    /* ── MESSAGES ── */
    .stSuccess {
        background-color: #f0fdf4 !important;
        border-left: 4px solid #22c55e !important;
        color: #166534 !important;
    }
    
    .stError {
        background-color: #fef2f2 !important;
        border-left: 4px solid #ef4444 !important;
        color: #991b1b !important;
    }
    
    .stWarning {
        background-color: #fffbeb !important;
        border-left: 4px solid #f59e0b !important;
        color: #92400e !important;
    }
    
    .stInfo {
        background-color: #ecf9ff !important;
        border-left: 4px solid #0ea5e9 !important;
        color: #0c2340 !important;
    }
    
    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 2px solid rgba(236, 72, 153, 0.15) !important;
        margin: 2rem 0 !important;
    }
    
    /* ── FOOTER ── */
    .footer-text {
        text-align: center;
        color: #be185d !important;
        font-size: 0.95rem;
        padding: 2.5rem;
        font-weight: 500;
        border-top: 2px solid rgba(236, 72, 153, 0.15);
    }
    
    /* ── HEADINGS ── */
    h1, h2, h3, h4, h5, h6 {
        color: #be185d !important;
        font-weight: 800 !important;
    }
    
    p, .stMarkdown p {
        color: #475569 !important;
        line-height: 1.7 !important;
        font-weight: 500;
    }
    
    /* ── UPLOAD BOX ── */
    .upload-box {
        border: 2px dashed #ec4899;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #fff5f7 0%, #ffe4ed 100%);
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #db2777;
        background: linear-gradient(135deg, #ffe4ed 0%, #ffb3d9 100%);
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
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
        'Excellent': '#10b981',
        'Good': '#0ea5e9',
        'Poor': '#f59e0b',
        'Very Poor': '#ef4444',
        'Unfit for Drinking': '#991b1b'
    }
    return colors.get(cat, '#ec4899')

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


# ── Top Navigation Buttons ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>💧 AquaVision</h1>
    <p>Water Quality Prediction Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Navigation
nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    btn_overview = st.button("📊 Overview", use_container_width=True, key="nav_overview")
with nav_col2:
    btn_predictor = st.button("🤖 Predictor", use_container_width=True, key="nav_predictor")
with nav_col3:
    btn_bulk = st.button("📤 Bulk Scanner", use_container_width=True, key="nav_bulk")

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "overview"

if btn_overview:
    st.session_state.current_page = "overview"
if btn_predictor:
    st.session_state.current_page = "predictor"
if btn_bulk:
    st.session_state.current_page = "bulk"

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "overview":
    st.markdown('<div class="section-title">📊 Overview</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-sub">observations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg WQI Score</div>
            <div class="metric-value">{df['WQI'].mean():.1f}</div>
            <div class="metric-sub">quality index</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">States Covered</div>
            <div class="metric-value">{df['state'].nunique()}</div>
            <div class="metric-sub">unique states</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stations</div>
            <div class="metric-value">{df['station_code'].nunique():,}</div>
            <div class="metric-sub">monitoring sites</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">Category Distribution</div>', unsafe_allow_html=True)
        cat_cnt = df['WQI_Category'].value_counts().reset_index()
        cat_cnt.columns = ['Category', 'Count']
        color_map = {'Excellent': '#10b981', 'Good': '#0ea5e9', 'Poor': '#f59e0b', 'Very Poor': '#ef4444', 'Unfit for Drinking': '#991b1b'}
        fig = px.pie(cat_cnt, names='Category', values='Count', color='Category', color_discrete_map=color_map, hole=0.4)
        fig.update_layout(template='plotly_white', height=350, font=dict(family="Poppins", size=12))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">Parameter Statistics</div>', unsafe_allow_html=True)
        params = ['temp', 'do', 'ph', 'conductivity', 'bod', 'nitrate']
        param_stats = pd.DataFrame({
            'Parameter': [p.upper() for p in params],
            'Mean': [df[p].mean() for p in params],
            'Std': [df[p].std() for p in params],
            'Min': [df[p].min() for p in params],
            'Max': [df[p].max() for p in params]
        }).round(2)
        st.dataframe(param_stats, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "predictor":
    st.markdown('<div class="section-title">🤖 WQI Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter water quality parameters to get instant WQI predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp    = st.number_input("🌡️ Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
        do      = st.number_input("💨 Dissolved Oxygen (mg/L)", 0.0, 20.0, 7.5, 0.1)
        ph      = st.number_input("⚗️ pH Level", 0.0, 14.0, 7.0, 0.1)
    
    with col2:
        cond    = st.number_input("⚡ Conductivity (µS/cm)", 0.0, 5000.0, 250.0, 1.0)
        bod     = st.number_input("🦠 BOD (mg/L)", 0.0, 100.0, 2.0, 0.1)
        nitrate = st.number_input("🧪 Nitrate (mg/L)", 0.0, 200.0, 5.0, 0.1)
    
    with col3:
        fecal   = st.number_input("🔬 Fecal Coliform", 0.0, 200000.0, 100.0, 1.0)
        total   = st.number_input("🔭 Total Coliform", 0.0, 200000.0, 500.0, 1.0)
    
    st.divider()
    
    if st.button("🚀 PREDICT WQI", use_container_width=True, type="primary"):
        row = {'temp':temp,'do':do,'ph':ph,'conductivity':cond,
               'bod':bod,'nitrate':nitrate,'fecal_coliform':fecal,'total_coliform':total}
        feat_scaled = scaler.transform([engineer_single(row)])
        wqi_pred = float(reg_model.predict(feat_scaled)[0])
        cat_pred = label_enc.inverse_transform(clf_model.predict(feat_scaled))[0]
        cat_color = wqi_color(cat_pred)
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted WQI Score</div>
                <div class="metric-value" style="color:{cat_color}">{wqi_pred:.2f}</div>
                <span class="wqi-badge" style="background:{cat_color};">{cat_pred}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=min(wqi_pred, 120),
                domain={'x':[0,1],'y':[0,1]}, title={'text':"WQI Gauge"},
                gauge={'axis':{'range':[0,120]},
                       'bar':{'color':cat_color},
                       'steps':[{'range':[0,25],'color':'rgba(16,185,129,0.2)'},
                                {'range':[25,50],'color':'rgba(14,165,233,0.2)'},
                                {'range':[50,75],'color':'rgba(245,158,11,0.2)'},
                                {'range':[75,100],'color':'rgba(239,68,68,0.2)'},
                                {'range':[100,120],'color':'rgba(153,27,27,0.2)'}]}))
            fig.update_layout(height=320, margin=dict(t=40,b=10), font=dict(family="Poppins", size=12))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BULK SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "bulk":
    st.markdown('<div class="section-title">📤 Bulk WQI Scanner</div>', unsafe_allow_html=True)
    st.markdown("Process multiple water samples at once")
    
    if 'bulk_results' not in st.session_state:
        st.session_state.bulk_results = None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.download_button("📄 Download Template", create_sample_csv(), "sample.csv", "text/csv", use_container_width=True):
            st.success("Template ready!")
    
    with col2:
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx', 'xls', 'json'], key='bulk_upload')
    
    with col3:
        if st.session_state.bulk_results is not None:
            csv_data = st.session_state.bulk_results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv_data, "results.csv", "text/csv", use_container_width=True)
        else:
            st.button("📥 Results (scan first)", disabled=True, use_container_width=True)
    
    st.divider()
    
    if uploaded_file is not None:
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
                
                if st.button("🚀 SCAN & PREDICT", use_container_width=True, type="primary"):
                    with st.spinner('⏳ Processing...'):
                        st.session_state.bulk_results = predict_batch_wqi(df_upload, reg_model, clf_model, scaler, label_enc)
                    st.success(f"✅ Done! {len(st.session_state.bulk_results)} records processed")
            else:
                st.error(f"❌ {validation_msg}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.bulk_results is not None:
        st.divider()
        st.markdown('<div class="section-title">📊 Results</div>', unsafe_allow_html=True)
        results_df = st.session_state.bulk_results.copy()
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        avg_wqi = pd.to_numeric(results_df['WQI_Score'], errors='coerce').mean()
        excellent_count = len(results_df[results_df['Category'] == 'Excellent'])
        poor_count = len(results_df[results_df['Category'].isin(['Poor', 'Very Poor', 'Unfit for Drinking'])])
        success_rate = len(results_df[results_df['Category'] != 'Processing Error']) / len(results_df) * 100
        
        col_stat1.metric("📊 Avg WQI", f"{avg_wqi:.2f}")
        col_stat2.metric("✅ Excellent", excellent_count)
        col_stat3.metric("⚠️ Needs Attention", poor_count)
        col_stat4.metric("✓ Success Rate", f"{success_rate:.0f}%")
        
        st.dataframe(results_df, use_container_width=True, height=400)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="footer-text">
    💧 <strong>AquaVision</strong> — Water Quality Intelligence Platform | 2024
</div>
""", unsafe_allow_html=True)
