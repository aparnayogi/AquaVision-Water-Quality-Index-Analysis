"""
💧 AquaVision — Water Quality Intelligence
Final Year Project | Streamlit Dashboard App
Run: streamlit run aquavision_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #F0F8FF; }
    .hero-banner {
        background: linear-gradient(135deg, #0D47A1 0%, #006064 50%, #1B5E20 100%);
        padding: 2.5rem 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(13,71,161,0.3);
    }
    .hero-banner h1 { font-size: 3rem; font-weight: 800; color: white; margin: 0; letter-spacing: -1px; }
    .hero-banner p  { font-size: 1.1rem; color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); border-left: 5px solid #1565C0; margin-bottom: 0.8rem;
    }
    .metric-card .label { font-size: 0.8rem; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #0D47A1; }
    .metric-card .sub   { font-size: 0.8rem; color: #888; }
    .wqi-badge {
        display: inline-block; padding: 0.4rem 1.2rem; border-radius: 50px;
        font-size: 1rem; font-weight: 700; color: white; margin-top: 0.5rem;
    }
    .predict-box {
        background: linear-gradient(135deg, #E3F2FD, #E8F5E9); border-radius: 16px;
        padding: 2rem; border: 2px solid #90CAF9; box-shadow: 0 4px 20px rgba(21,101,192,0.15);
    }
    .section-title {
        font-size: 1.4rem; font-weight: 700; color: #0D47A1;
        margin: 1.5rem 0 1rem; padding-left: 0.8rem; border-left: 4px solid #1565C0;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D47A1 0%, #006064 100%); }
    div[data-testid="stSidebar"] * { color: white !important; }
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
    for cat, col in [('< 25 — Excellent','#1B5E20'),('25–50 — Good','#4CAF50'),
                      ('50–75 — Poor','#FF9800'),('75–100 — Very Poor','#F44336'),
                      ('> 100 — Unfit','#880E4F')]:
        st.markdown(f"<span style='color:{col};font-weight:600'>■</span> {cat}", unsafe_allow_html=True)

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

    st.markdown('<div class="section-title">WQI Category Breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cat_cnt = dff['WQI_Category'].value_counts().reset_index()
        cat_cnt.columns = ['Category', 'Count']
        fig = px.pie(cat_cnt, names='Category', values='Count',
                     color='Category', color_discrete_map=CAT_COLOR_MAP,
                     title='Distribution of WQI Categories', hole=0.42)
        fig.update_traces(textinfo='label+percent', pull=[0.04]*len(cat_cnt))
        fig.update_layout(template='plotly_white', showlegend=True, height=380)
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
                          yaxis_title='WQI Score', template='plotly_white', height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Parameter Overview</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, col_name in enumerate(['temp','do','ph','conductivity','bod','nitrate']):
        cols[i].metric(col_name.upper(), f"{dff[col_name].mean():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Exploratory Analysis":
    st.markdown('<div class="section-title">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔥 Correlations", "📦 Boxplots", "🔵 PCA"])
    num_cols_list = ['temp','do','ph','conductivity','bod','nitrate','fecal_coliform','total_coliform']

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
        fig.update_layout(template='plotly_white', height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr_matrix = dff[num_cols_list + ['WQI']].corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title='Feature Correlation Heatmap')
        fig.update_layout(template='plotly_white', height=520)
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
                          title_text='Parameter Distribution by WQI Category')
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
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        with col_pca2:
            ev = pca.explained_variance_ratio_ * 100
            fig = go.Figure(go.Bar(x=[f'PC{i+1}' for i in range(3)], y=ev,
                                   marker_color=['#0D47A1','#1565C0','#42A5F5'],
                                   text=[f'{v:.1f}%' for v in ev], textposition='outside'))
            fig.update_layout(title='Explained Variance', template='plotly_white', height=300)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends & Patterns":
    st.markdown('<div class="section-title">📈 Trends & Pattern Analysis</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["⏳ Time Trends", "🌊 Parameter Deep-Dive"])

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
                          xaxis_title='Year', template='plotly_white', height=450)
        st.plotly_chart(fig, use_container_width=True)

        cat_yr = dff.groupby(['year','WQI_Category']).size().reset_index(name='count')
        fig2 = px.area(cat_yr, x='year', y='count', color='WQI_Category',
                       color_discrete_map=CAT_COLOR_MAP, title='WQI Category Composition Over Years')
        fig2.update_layout(template='plotly_white', height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        param_dd = st.selectbox("Parameter", ['temp','do','ph','conductivity','bod','nitrate'])
        top_n    = st.slider("Top N States", 5, 20, 10)
        state_param = dff.groupby('state')[param_dd].mean().sort_values(ascending=False).head(top_n)
        fig = px.bar(state_param.reset_index(), x='state', y=param_dd,
                     title=f'Top {top_n} States by avg {param_dd.upper()}',
                     color=param_dd, color_continuous_scale='Blues', text_auto='.2f')
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 WQI Predictor":
    st.markdown('<div class="section-title">🤖 AI-Powered WQI Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter water parameter values below to predict the Water Quality Index:")
    st.markdown('<div class="predict-box">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        temp    = st.number_input("🌡️ Temperature (°C)",           0.0, 50.0,     25.0, 0.1)
        do      = st.number_input("💨 Dissolved Oxygen (mg/L)",    0.0, 20.0,      7.5, 0.1)
        ph      = st.number_input("⚗️ pH",                         0.0, 14.0,      7.0, 0.1)
    with col2:
        cond    = st.number_input("⚡ Conductivity (µmhos/cm)",    0.0, 5000.0,  250.0, 1.0)
        bod     = st.number_input("🦠 BOD (mg/L)",                 0.0, 100.0,     2.0, 0.1)
        nitrate = st.number_input("🧪 Nitrate (mg/L)",             0.0, 200.0,     5.0, 0.1)
    with col3:
        fecal   = st.number_input("🔬 Fecal Coliform (MPN/100mL)", 0.0, 200000.0,100.0, 1.0)
        total   = st.number_input("🔬 Total Coliform (MPN/100mL)", 0.0, 200000.0,500.0, 1.0)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Predict Water Quality Index", use_container_width=True, type="primary"):
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
            fig.update_layout(height=250, margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)
        with col_r3:
            probs = pd.DataFrame({'Category':label_enc.classes_,
                                  'Probability':cat_prob*100}).sort_values('Probability', ascending=True)
            fig = px.bar(probs, x='Probability', y='Category', orientation='h',
                         title='Category Confidence (%)', color='Probability',
                         color_continuous_scale='Blues', text_auto='.1f')
            fig.update_layout(height=250, template='plotly_white',
                              margin=dict(t=40,b=10), showlegend=False)
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
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ State Intelligence":
    st.markdown('<div class="section-title">🗺️ State-wise Water Quality Intelligence</div>', unsafe_allow_html=True)

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
                          xaxis={'categoryorder':'total descending'}, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.treemap(state_stats, path=['state'], values='records',
                          color='avg_wqi', color_continuous_scale='RdYlGn_r',
                          title='State Coverage', hover_data=['avg_wqi','category'])
        fig2.update_layout(height=450)
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
                             template='plotly_white', height=420)
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
<div style='text-align:center; color:#888; font-size:0.85rem; padding:1rem'>
    💧 <strong>AquaVision</strong> — Water Quality Intelligence Platform &nbsp;|&nbsp;
    Final Year Project &nbsp;|&nbsp; Built with Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
