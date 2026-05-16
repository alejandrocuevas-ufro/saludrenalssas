"""
DASHBOARD INTERACTIVO — Estudio de prevalencia de ERC
Región de la Araucanía, Chile

Para correr:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ERC Araucanía — Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta
C_MAIN = '#2E5266'
C_ACC = '#D9594C'
C_LIGHT = '#9FC2BA'
C_GRAY = '#888888'
C_OK = '#5B8C5A'
C_PPOO = '#A06D3F'
C_NOPP = '#658a93'

# CSS personalizado
st.markdown("""
<style>
    .stMetric { background: #f8f9fa; padding: 10px; border-radius: 6px; border-left: 4px solid #2E5266; }
    h1 { color: #2E5266; }
    h2 { color: #2E5266; border-bottom: 2px solid #9FC2BA; padding-bottom: 4px; }
    h3 { color: #2E5266; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { background: #f0f2f5; padding: 8px 16px; border-radius: 4px; }
    .stTabs [aria-selected="true"] { background: #2E5266 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# CARGA DE DATOS Y PREPARACIÓN
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_data(path):
    df = pd.read_excel(path, sheet_name='Datos_analisis')

    # Grupos etarios
    def age_grp(a):
        if pd.isna(a): return np.nan
        if a < 30: return '<30'
        elif a < 45: return '30-44'
        elif a < 60: return '45-59'
        elif a < 75: return '60-74'
        else: return '≥75'
    df['Age_group'] = df['Age'].apply(age_grp)

    df['Sex_lbl'] = df['Sex'].map({0: 'Mujer', 1: 'Hombre'})
    df['PPOO_lbl'] = df['¿PPOO?'].map({0: 'No PPOO', 1: 'PPOO'})

    # Limpieza adicional ya validada
    df.loc[(df['Height_cm'] < 3) & df['Height_cm'].notna(), 'Height_cm'] *= 100
    df.loc[(df['Height_cm'] < 100) | (df['Height_cm'] > 220), 'Height_cm'] = np.nan
    df.loc[(df['Weight_kg'] > 300) | (df['Weight_kg'] < 25), 'Weight_kg'] = np.nan
    df['BMI'] = df['Weight_kg'] / (df['Height_cm']/100)**2
    df.loc[(df['BMI']<10)|(df['BMI']>70), 'BMI'] = np.nan
    for col in ['SBP_1st','SBP_2nd','DBP_1st','DBP_2nd']:
        df.loc[(df[col]>250)|(df[col]<40), col] = np.nan

    def weight_cat(bmi):
        if pd.isna(bmi): return np.nan
        if bmi < 18.5: return 'bajo peso'
        elif bmi < 25: return 'peso normal'
        elif bmi < 30: return 'sobrepeso'
        else: return 'obeso'
    df['Weight_Category_clean'] = df['BMI'].apply(weight_cat)
    df['Obesity'] = (df['Weight_Category_clean']=='obeso').astype('Int64')
    df.loc[df['Weight_Category_clean'].isna(), 'Obesity'] = pd.NA

    # CKD definitions
    df['CKD_eGFR_lt60'] = (df['eGFR_1st'] < 60).astype('Int64')
    df.loc[df['eGFR_1st'].isna(), 'CKD_eGFR_lt60'] = pd.NA

    def alb_cat(p):
        if pd.isna(p): return np.nan
        if p == 'Neg': return 'A1 (Neg)'
        try:
            pv = float(p)
            return 'A2 (30 mg/dL)' if pv <= 30 else 'A3 (≥100 mg/dL)'
        except: return np.nan
    df['Albuminuria_cat'] = df['Proteinuria'].apply(alb_cat)
    df['Proteinuria_pos'] = df['Proteinuria'].apply(
        lambda x: 0 if x=='Neg' else (1 if pd.notna(x) else pd.NA)
    ).astype('Int64')
    df['Hematuria_pos'] = df['Blood'].apply(
        lambda x: 0 if x=='Neg' else (1 if pd.notna(x) else pd.NA)
    ).astype('Int64')

    df['CKD_extended'] = ((df['eGFR_1st']<60) | (df['Proteinuria_pos']==1)).astype('Int64')
    df.loc[df['eGFR_1st'].isna() & df['Proteinuria_pos'].isna(), 'CKD_extended'] = pd.NA

    df['HTA_measured'] = df['BP_Final'].isin(
        ['Stage 1 HTN','Stage 2 HTN','Hypertensive Crisis']
    ).astype('Int64')
    df.loc[df['BP_Final'].isin(['Missing data', np.nan]), 'HTA_measured'] = pd.NA

    df['Glucose_high'] = (df['Glucose_1st']>=200).astype('Int64')
    df.loc[df['Glucose_1st'].isna(), 'Glucose_high'] = pd.NA

    # KDIGO risk
    def kdigo_risk(g, a):
        risk_map = {
            ('G1','A1'):'Bajo',('G1','A2'):'Moderado',('G1','A3'):'Alto',
            ('G2','A1'):'Bajo',('G2','A2'):'Moderado',('G2','A3'):'Alto',
            ('G3a','A1'):'Moderado',('G3a','A2'):'Alto',('G3a','A3'):'Muy alto',
            ('G3b','A1'):'Alto',('G3b','A2'):'Muy alto',('G3b','A3'):'Muy alto',
            ('G4','A1'):'Muy alto',('G4','A2'):'Muy alto',('G4','A3'):'Muy alto',
            ('G5','A1'):'Muy alto',('G5','A2'):'Muy alto',('G5','A3'):'Muy alto',
        }
        if pd.isna(g) or pd.isna(a): return np.nan
        return risk_map.get((g, a.split(' ')[0]), np.nan)
    df['KDIGO_risk'] = df.apply(lambda r: kdigo_risk(r['CKD_KDIGO_G'], r['Albuminuria_cat']), axis=1)

    return df


# ═══════════════════════════════════════════════════════════
# HELPERS ESTADÍSTICOS
# ═══════════════════════════════════════════════════════════
def prev_ci_wilson(s, t):
    if t == 0:
        return (np.nan, np.nan, np.nan)
    p = s/t
    z = 1.96
    den = 1 + z**2/t
    centre = p + z**2/(2*t)
    margin = z*np.sqrt(p*(1-p)/t + z**2/(4*t**2))
    return (p*100, (centre-margin)/den*100, (centre+margin)/den*100)


def chi2_pvalue(df, var, outcome):
    sub = df.dropna(subset=[var, outcome])
    if len(sub) == 0 or sub[var].nunique() < 2:
        return np.nan
    try:
        tab = pd.crosstab(sub[var], sub[outcome])
        return stats.chi2_contingency(tab)[1]
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════
# SIDEBAR — Filtros
# ═══════════════════════════════════════════════════════════
st.sidebar.title("🩺 Filtros del análisis")
st.sidebar.caption("Aplican a todas las pestañas")

# Carga
uploaded = st.sidebar.file_uploader(
    "Archivo de datos (.xlsx)",
    type=['xlsx'],
    help="Si no subes archivo, se buscará CKD_DATA_v2.xlsx en el directorio actual"
)

if uploaded is not None:
    df_raw = load_data(uploaded)
else:
    try:
        df_raw = load_data('CKD_DATA_v2.xlsx')
    except Exception as e:
        st.error(f"No se encontró CKD_DATA_v2.xlsx en este directorio y no se subió archivo.\n\nError: {e}")
        st.stop()

# Filtros
st.sidebar.markdown("### Filtros")

age_min, age_max = st.sidebar.slider(
    "Rango de edad (años)",
    int(df_raw['Age'].min()) if df_raw['Age'].notna().any() else 18,
    int(df_raw['Age'].max()) if df_raw['Age'].notna().any() else 100,
    (int(df_raw['Age'].min()) if df_raw['Age'].notna().any() else 18,
     int(df_raw['Age'].max()) if df_raw['Age'].notna().any() else 100),
)

# Comunas con n≥10
comm_counts = df_raw['Community'].value_counts()
all_comms = sorted(comm_counts[comm_counts >= 5].index.tolist())
selected_comms = st.sidebar.multiselect(
    "Comunas (solo con n ≥ 5)",
    options=all_comms,
    default=all_comms,
    help="Por defecto todas. Quita las que no quieras incluir."
)

ppoo_filter = st.sidebar.multiselect(
    "Pueblo originario",
    options=['PPOO', 'No PPOO'],
    default=['PPOO', 'No PPOO'],
)

sex_filter = st.sidebar.multiselect(
    "Sexo",
    options=['Mujer', 'Hombre'],
    default=['Mujer', 'Hombre'],
)

st.sidebar.markdown("### Filtros opcionales")

bmi_filter = st.sidebar.multiselect(
    "Estado nutricional (IMC)",
    options=['bajo peso','peso normal','sobrepeso','obeso'],
    default=['bajo peso','peso normal','sobrepeso','obeso'],
)

hta_filter = st.sidebar.radio(
    "HTA autoreportada",
    options=['Todos','Solo HTA conocida','Solo sin HTA conocida'],
    index=0,
)

dm_filter = st.sidebar.radio(
    "DM autoreportada",
    options=['Todos','Solo DM conocida','Solo sin DM conocida'],
    index=0,
)

# Aplicar filtros
df = df_raw[
    (df_raw['Age'].between(age_min, age_max)) &
    (df_raw['Community'].isin(selected_comms)) &
    (df_raw['PPOO_lbl'].isin(ppoo_filter)) &
    (df_raw['Sex_lbl'].isin(sex_filter)) &
    (df_raw['Weight_Category_clean'].isin(bmi_filter) | df_raw['Weight_Category_clean'].isna())
].copy()

if hta_filter == 'Solo HTA conocida':
    df = df[df['LE HAN DICHO QUE TIENE HTA'] == 1]
elif hta_filter == 'Solo sin HTA conocida':
    df = df[df['LE HAN DICHO QUE TIENE HTA'] == 0]

if dm_filter == 'Solo DM conocida':
    df = df[df['LE HAN DICHO QUE TIENE DM?'] == 1]
elif dm_filter == 'Solo sin DM conocida':
    df = df[df['LE HAN DICHO QUE TIENE DM?'] == 0]

st.sidebar.markdown("---")
st.sidebar.metric("N tras filtros", f"{len(df):,}", f"{len(df)/len(df_raw)*100:.1f}% del total")

if len(df) < 20:
    st.warning(f"⚠️ Muestra muy pequeña ({len(df)} registros). Los resultados serán inestables.")

# ═══════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ═══════════════════════════════════════════════════════════
st.title("Estudio de prevalencia de ERC — Región de la Araucanía")
st.caption("Dashboard interactivo · Muestra de screening en operativos comunitarios")

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab_overview, tab_prev, tab_strat, tab_subgrp, tab_risk, tab_score, tab_concord, tab_data = st.tabs([
    "🏠 Resumen", "📊 Prevalencia", "🔬 Estratificación",
    "👥 Subgrupos", "⚠️ Factores riesgo", "📈 Score de riesgo",
    "🔄 Concordancia", "📋 Datos"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ═══════════════════════════════════════════════════════════
with tab_overview:
    st.header("Características de la muestra")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N total", f"{len(df):,}")
    edad_media = df['Age'].mean()
    c2.metric("Edad media (años)", f"{edad_media:.1f}" if pd.notna(edad_media) else "—")
    pct_m = (df['Sex']==1).sum()/df['Sex'].notna().sum()*100 if df['Sex'].notna().any() else 0
    c3.metric("% Hombres", f"{pct_m:.1f}%")
    pct_ppoo = (df['¿PPOO?']==1).sum()/df['¿PPOO?'].notna().sum()*100 if df['¿PPOO?'].notna().any() else 0
    c4.metric("% PPOO", f"{pct_ppoo:.1f}%")

    st.markdown("### Prevalencias principales")
    c1, c2, c3, c4 = st.columns(4)

    ckd_n = (df['CKD_eGFR_lt60']==1).sum()
    ckd_N = df['CKD_eGFR_lt60'].notna().sum()
    p, l, h = prev_ci_wilson(ckd_n, ckd_N)
    c1.metric("ERC (eGFR<60)", f"{p:.1f}%" if pd.notna(p) else "—",
              f"IC95% {l:.1f}–{h:.1f}" if pd.notna(l) else "")

    ext_n = (df['CKD_extended']==1).sum()
    ext_N = df['CKD_extended'].notna().sum()
    p2, l2, h2 = prev_ci_wilson(ext_n, ext_N)
    c2.metric("ERC ampliada", f"{p2:.1f}%" if pd.notna(p2) else "—",
              f"IC95% {l2:.1f}–{h2:.1f}" if pd.notna(l2) else "",
              help="eGFR<60 O proteinuria positiva")

    hta_n = (df['HTA_measured']==1).sum()
    hta_N = df['HTA_measured'].notna().sum()
    p3, l3, h3 = prev_ci_wilson(hta_n, hta_N)
    c3.metric("HTA medida", f"{p3:.1f}%" if pd.notna(p3) else "—",
              f"IC95% {l3:.1f}–{h3:.1f}" if pd.notna(l3) else "")

    ob_n = (df['Obesity']==1).sum()
    ob_N = df['Obesity'].notna().sum()
    p4, l4, h4 = prev_ci_wilson(ob_n, ob_N)
    c4.metric("Obesidad", f"{p4:.1f}%" if pd.notna(p4) else "—",
              f"IC95% {l4:.1f}–{h4:.1f}" if pd.notna(l4) else "")

    st.markdown("### Pirámide poblacional")
    pir_df = df.dropna(subset=['Age_group','Sex_lbl']).copy()
    if len(pir_df):
        order = ['<30','30-44','45-59','60-74','≥75']
        pir = pd.crosstab(pir_df['Age_group'], pir_df['Sex_lbl']).reindex(order).fillna(0)

        fig = go.Figure()
        if 'Hombre' in pir.columns:
            fig.add_trace(go.Bar(
                y=pir.index, x=-pir.get('Hombre', 0),
                name='Hombres', orientation='h', marker_color=C_MAIN,
                hovertemplate='Hombres: %{customdata}<extra></extra>',
                customdata=pir.get('Hombre', 0)
            ))
        if 'Mujer' in pir.columns:
            fig.add_trace(go.Bar(
                y=pir.index, x=pir.get('Mujer', 0),
                name='Mujeres', orientation='h', marker_color=C_ACC,
                hovertemplate='Mujeres: %{x}<extra></extra>',
            ))
        max_v = max(pir.max().max(), 1)
        fig.update_layout(
            barmode='relative',
            xaxis=dict(
                title='Cantidad',
                tickvals=[-max_v, -max_v/2, 0, max_v/2, max_v],
                ticktext=[str(int(max_v)), str(int(max_v/2)), '0', str(int(max_v/2)), str(int(max_v))]
            ),
            yaxis=dict(title='Grupo etario'),
            height=400, margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distribución de eGFR")
    if df['eGFR_1st'].notna().any():
        fig = px.histogram(
            df.dropna(subset=['eGFR_1st']),
            x='eGFR_1st', nbins=40, color_discrete_sequence=[C_MAIN]
        )
        fig.add_vline(x=60, line_dash="dash", line_color=C_ACC, annotation_text="60 mL/min/1.73m²")
        fig.add_vline(x=90, line_dash="dot", line_color=C_GRAY)
        fig.update_layout(
            xaxis_title='eGFR (mL/min/1.73m²)',
            yaxis_title='Frecuencia',
            height=350, margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — PREVALENCIA
# ═══════════════════════════════════════════════════════════
with tab_prev:
    st.header("Prevalencia de ERC — definiciones operacionales")

    rows = []
    for label, var in [
        ("ERC estricta (eGFR<60 mL/min/1.73m²)", 'CKD_eGFR_lt60'),
        ("ERC ampliada (eGFR<60 O proteinuria+)", 'CKD_extended'),
        ("Proteinuria positiva aislada", 'Proteinuria_pos'),
        ("Hematuria positiva aislada", 'Hematuria_pos'),
    ]:
        n = (df[var]==1).sum()
        N = df[var].notna().sum()
        p, l, h = prev_ci_wilson(n, N)
        rows.append([label, n, N, f"{p:.1f}%" if pd.notna(p) else "—",
                     f"{l:.1f}–{h:.1f}" if pd.notna(l) else "—"])
    prev_df = pd.DataFrame(rows, columns=['Definición', 'n', 'N', 'Prevalencia', 'IC95%'])
    st.dataframe(prev_df, use_container_width=True, hide_index=True)

    st.markdown("### Distribución por estadio KDIGO")
    c1, c2 = st.columns([2, 1])

    with c1:
        if df['CKD_KDIGO_G'].notna().any():
            order = ['G1','G2','G3a','G3b','G4','G5']
            counts = df['CKD_KDIGO_G'].value_counts().reindex(order).fillna(0)
            total = counts.sum()
            pcts = counts/total*100 if total else counts
            colors = [C_OK, C_LIGHT, '#F4E04D', '#F08C5A', C_ACC, '#7A1818']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=order, y=counts.values, marker_color=colors,
                text=[f'{int(c)}<br>({p:.1f}%)' for c, p in zip(counts, pcts)],
                textposition='outside'
            ))
            fig.update_layout(
                xaxis_title='Estadio KDIGO G',
                yaxis_title='Número de pacientes',
                showlegend=False, height=400,
                margin=dict(l=10, r=10, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Riesgo combinado KDIGO**")
        if df['KDIGO_risk'].notna().any():
            risk_counts = df['KDIGO_risk'].value_counts().reindex(
                ['Bajo','Moderado','Alto','Muy alto']
            ).fillna(0)
            total = risk_counts.sum()
            risk_colors_d = {'Bajo': C_OK, 'Moderado':'#F4E04D','Alto':'#F08C5A','Muy alto':C_ACC}

            fig = go.Figure(go.Pie(
                labels=risk_counts.index, values=risk_counts.values,
                marker=dict(colors=[risk_colors_d[k] for k in risk_counts.index]),
                hole=0.4,
                textinfo='label+percent', textposition='outside'
            ))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            hi = ((df['KDIGO_risk']=='Alto') | (df['KDIGO_risk']=='Muy alto')).sum()
            st.metric("Alto + Muy alto", f"{hi}", f"{hi/total*100:.1f}% del total" if total else "")

    st.markdown("### Matriz KDIGO (eGFR × Albuminuria)")
    sub = df.dropna(subset=['CKD_KDIGO_G', 'Albuminuria_cat'])
    if len(sub) > 0:
        tab = pd.crosstab(sub['CKD_KDIGO_G'], sub['Albuminuria_cat'])
        tab = tab.reindex(index=['G1','G2','G3a','G3b','G4','G5']).reindex(
            columns=['A1 (Neg)','A2 (30 mg/dL)','A3 (≥100 mg/dL)']
        ).fillna(0).astype(int)

        risk_grid = pd.DataFrame([
            ['Bajo','Moderado','Alto'],
            ['Bajo','Moderado','Alto'],
            ['Moderado','Alto','Muy alto'],
            ['Alto','Muy alto','Muy alto'],
            ['Muy alto','Muy alto','Muy alto'],
            ['Muy alto','Muy alto','Muy alto'],
        ], index=tab.index, columns=tab.columns)

        risk_num = risk_grid.map({'Bajo':1,'Moderado':2,'Alto':3,'Muy alto':4}.get)

        fig = go.Figure(go.Heatmap(
            z=risk_num.values, x=tab.columns, y=tab.index,
            colorscale=[[0,'#A8D5A0'],[0.33,'#F4E04D'],[0.66,'#F08C5A'],[1,'#C53030']],
            showscale=False,
            text=tab.values.astype(str), texttemplate='%{text}',
            textfont=dict(size=14, color='white'),
            hovertemplate='Estadio %{y} - %{x}: %{text}<extra></extra>'
        ))
        fig.update_layout(
            xaxis_title='Albuminuria', yaxis_title='Estadio eGFR',
            yaxis=dict(autorange='reversed'),
            height=380, margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — ESTRATIFICACIÓN
# ═══════════════════════════════════════════════════════════
with tab_strat:
    st.header("Prevalencia estratificada")

    col1, col2 = st.columns([1, 3])
    with col1:
        var_strat = st.selectbox(
            "Variable de estratificación",
            options=[
                'Age_group', 'Sex_lbl', 'PPOO_lbl', 'Community',
                'Weight_Category_clean', 'BP_Final',
                'LE HAN DICHO QUE TIENE HTA', 'LE HAN DICHO QUE TIENE DM?',
                'ANTEC. FLIAR ERC', 'TABACO',
            ],
            format_func=lambda x: {
                'Age_group': 'Grupo etario',
                'Sex_lbl': 'Sexo',
                'PPOO_lbl': 'Pueblo originario',
                'Community': 'Comuna',
                'Weight_Category_clean': 'Estado nutricional',
                'BP_Final': 'Categoría PA',
                'LE HAN DICHO QUE TIENE HTA': 'HTA autoreportada',
                'LE HAN DICHO QUE TIENE DM?': 'DM autoreportada',
                'ANTEC. FLIAR ERC': 'Antecedente familiar ERC',
                'TABACO': 'Tabaquismo',
            }.get(x, x)
        )
        outcome = st.radio(
            "Definición de ERC",
            options=['CKD_eGFR_lt60', 'CKD_extended'],
            format_func=lambda x: 'Estricta (eGFR<60)' if x=='CKD_eGFR_lt60' else 'Ampliada (incluye proteinuria)',
        )
        min_n = st.number_input("N mínimo por estrato", 1, 100, 5)

    with col2:
        sub = df.dropna(subset=[var_strat, outcome])
        if len(sub) == 0:
            st.warning("Sin datos disponibles para esta variable con los filtros actuales.")
        else:
            grp = sub.groupby(var_strat).agg(n_ckd=(outcome, 'sum'), N=(outcome, 'size'))
            grp = grp[grp['N'] >= min_n]
            grp['prev'] = grp['n_ckd']/grp['N']*100
            grp[['ci_lo','ci_hi']] = grp.apply(
                lambda r: pd.Series(prev_ci_wilson(r['n_ckd'], r['N'])[1:]), axis=1
            )

            if var_strat == 'Age_group':
                grp = grp.reindex(['<30','30-44','45-59','60-74','≥75']).dropna()
            elif var_strat == 'Weight_Category_clean':
                grp = grp.reindex(['bajo peso','peso normal','sobrepeso','obeso']).dropna()
            else:
                grp = grp.sort_values('prev', ascending=False)

            grp = grp.reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=grp[var_strat].astype(str), y=grp['prev'],
                error_y=dict(type='data',
                             array=grp['ci_hi']-grp['prev'],
                             arrayminus=grp['prev']-grp['ci_lo']),
                marker_color=C_MAIN,
                text=[f'{p:.1f}%<br>(n={int(n)})' for p, n in zip(grp['prev'], grp['N'])],
                textposition='outside'
            ))
            fig.update_layout(
                yaxis_title='Prevalencia ERC (%)',
                xaxis_title=var_strat,
                height=400, margin=dict(l=10, r=10, t=30, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            p_chi = chi2_pvalue(sub, var_strat, outcome)
            if pd.notna(p_chi):
                st.caption(f"**Test chi² entre estratos: p = {p_chi:.4f}** "
                           f"{'(significativo)' if p_chi<0.05 else '(no significativo)'}")

            tab_show = grp[[var_strat,'n_ckd','N','prev','ci_lo','ci_hi']].copy()
            tab_show.columns = ['Estrato','n ERC','N','Prevalencia %','IC95% inf','IC95% sup']
            tab_show['Prevalencia %'] = tab_show['Prevalencia %'].round(1)
            tab_show['IC95% inf'] = tab_show['IC95% inf'].round(1)
            tab_show['IC95% sup'] = tab_show['IC95% sup'].round(1)
            st.dataframe(tab_show, use_container_width=True, hide_index=True)

    # Comparación PPOO × edad
    st.markdown("---")
    st.markdown("### Comparación PPOO × edad")
    sub = df.dropna(subset=['Age_group','PPOO_lbl', outcome])
    if len(sub) > 0:
        rows = []
        for g in ['<30','30-44','45-59','60-74','≥75']:
            for ppoo in ['PPOO','No PPOO']:
                s = sub[(sub['Age_group']==g) & (sub['PPOO_lbl']==ppoo)]
                if len(s) > 0:
                    n = (s[outcome]==1).sum()
                    p, lo, hi = prev_ci_wilson(n, len(s))
                    rows.append({'Edad': g, 'PPOO': ppoo, 'prev': p, 'lo': lo, 'hi': hi, 'N': len(s)})
        plot_df = pd.DataFrame(rows)

        if len(plot_df) > 0:
            fig = px.bar(
                plot_df, x='Edad', y='prev', color='PPOO', barmode='group',
                color_discrete_map={'PPOO': C_PPOO, 'No PPOO': C_NOPP},
                error_y=plot_df['hi']-plot_df['prev'],
                category_orders={'Edad': ['<30','30-44','45-59','60-74','≥75']},
                hover_data=['N']
            )
            fig.update_layout(
                yaxis_title='Prevalencia ERC (%)',
                xaxis_title='Grupo etario',
                height=400, margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — SUBGRUPOS
# ═══════════════════════════════════════════════════════════
with tab_subgrp:
    st.header("Subgrupos críticos")

    # Score de carga acumulada de factores de riesgo
    st.markdown("### Carga acumulada de factores de riesgo")
    st.caption("Factores: edad≥60, HTA medida, DM (autoreporte o HGT>200), obesidad, antec familiar ERC")

    rf_vars = {
        'rf_age':  (df['Age']>=60).astype(float),
        'rf_hta':  (df['HTA_measured']==1).astype(float).where(df['HTA_measured'].notna()),
        'rf_dm':   ((df['LE HAN DICHO QUE TIENE DM?']==1) | (df['Glucose_high']==1)).astype(float),
        'rf_ob':   (df['Obesity']==1).astype(float).where(df['Obesity'].notna()),
        'rf_fam':  (df['ANTEC. FLIAR ERC']==1).astype(float).where(df['ANTEC. FLIAR ERC'].notna()),
    }
    df['RiskScore_n'] = pd.DataFrame(rf_vars).sum(axis=1, min_count=3)
    df['RiskScore_lbl'] = pd.cut(df['RiskScore_n'], bins=[-1,0,1,2,5], labels=['0','1','2','≥3'])

    sub = df.dropna(subset=['CKD_eGFR_lt60','RiskScore_lbl'])
    if len(sub) > 0:
        rows = []
        for s in ['0','1','2','≥3']:
            sg = sub[sub['RiskScore_lbl']==s]
            if len(sg):
                n = (sg['CKD_eGFR_lt60']==1).sum()
                p, lo, hi = prev_ci_wilson(n, len(sg))
                rows.append({'Score': s, 'prev': p, 'lo': lo, 'hi': hi, 'N': len(sg), 'n_ckd': n})
        sc = pd.DataFrame(rows)

        if len(sc) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sc['Score'], y=sc['prev'],
                error_y=dict(type='data', array=sc['hi']-sc['prev'], arrayminus=sc['prev']-sc['lo']),
                marker_color=[C_LIGHT,'#7FA89C',C_MAIN,C_ACC],
                text=[f'{p:.1f}%<br>(n={N})' for p, N in zip(sc['prev'], sc['N'])],
                textposition='outside'
            ))
            fig.update_layout(
                yaxis_title='Prevalencia ERC (%)',
                xaxis_title='N° factores de riesgo presentes',
                height=400, margin=dict(l=10, r=10, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Combinaciones clínicas críticas")
    combos = [
        ('Sin factores conocidos (<60 sin HTA/DM/obesidad)',
         (df['Age']<60) & (df['LE HAN DICHO QUE TIENE HTA']==0) &
         (df['LE HAN DICHO QUE TIENE DM?']==0) & (df['Obesity']==0)),
        ('Edad ≥60 sin HTA/DM/obesidad',
         (df['Age']>=60) & (df['LE HAN DICHO QUE TIENE HTA']==0) &
         (df['LE HAN DICHO QUE TIENE DM?']==0) & (df['Obesity']==0)),
        ('DM + Obesidad',
         (df['LE HAN DICHO QUE TIENE DM?']==1) & (df['Obesity']==1)),
        ('HTA + Obesidad',
         (df['LE HAN DICHO QUE TIENE HTA']==1) & (df['Obesity']==1)),
        ('HTA + DM (autoreporte)',
         (df['LE HAN DICHO QUE TIENE HTA']==1) & (df['LE HAN DICHO QUE TIENE DM?']==1)),
        ('HTA + DM + Obesidad',
         (df['LE HAN DICHO QUE TIENE HTA']==1) &
         (df['LE HAN DICHO QUE TIENE DM?']==1) & (df['Obesity']==1)),
        ('Edad≥60 + HTA + DM',
         (df['Age']>=60) & (df['LE HAN DICHO QUE TIENE HTA']==1) &
         (df['LE HAN DICHO QUE TIENE DM?']==1)),
        ('Antec familiar + Proteinuria+',
         (df['ANTEC. FLIAR ERC']==1) & (df['Proteinuria_pos']==1)),
    ]
    rows = []
    for lbl, mask in combos:
        s = df[mask.fillna(False)].dropna(subset=['CKD_eGFR_lt60'])
        n = (s['CKD_eGFR_lt60']==1).sum()
        N = len(s)
        if N >= 5:
            p, lo, hi = prev_ci_wilson(n, N)
            rows.append({'Combinación': lbl, 'N': N, 'n_ERC': n,
                         'Prevalencia': p, 'lo': lo, 'hi': hi})
    if rows:
        cdf = pd.DataFrame(rows).sort_values('Prevalencia')
        fig = go.Figure(go.Bar(
            y=cdf['Combinación'], x=cdf['Prevalencia'],
            orientation='h', marker_color=C_LIGHT,
            error_x=dict(type='data', array=cdf['hi']-cdf['Prevalencia'],
                         arrayminus=cdf['Prevalencia']-cdf['lo']),
            text=[f'{p:.1f}% (n={n})' for p, n in zip(cdf['Prevalencia'], cdf['N'])],
            textposition='outside'
        ))
        prev_global, _, _ = prev_ci_wilson((df['CKD_eGFR_lt60']==1).sum(),
                                           df['CKD_eGFR_lt60'].notna().sum())
        if pd.notna(prev_global):
            fig.add_vline(x=prev_global, line_dash='dash', line_color='black',
                          annotation_text=f"Global {prev_global:.1f}%")
        fig.update_layout(
            xaxis_title='Prevalencia ERC (%)', yaxis_title='',
            height=450, margin=dict(l=10, r=80, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 5 — FACTORES DE RIESGO
# ═══════════════════════════════════════════════════════════
with tab_risk:
    st.header("Factores de riesgo")

    st.markdown("### RR crudo por factor (eGFR<60)")
    risk_factors = [
        ('LE HAN DICHO QUE TIENE HTA', 'HTA conocida (autoreporte)'),
        ('LE HAN DICHO QUE TIENE DM?', 'DM conocida (autoreporte)'),
        ('HTA_measured', 'HTA medida (PA≥130/80)'),
        ('Glucose_high', 'Glicemia >200'),
        ('Obesity', 'Obesidad (IMC≥30)'),
        ('HISTORIA PERSONAL LITIASIS', 'Litiasis previa'),
        ('ITU RECURRENTE', 'ITU recurrente'),
        ('ANTEC. FLIAR ERC', 'Antec familiar ERC'),
        ('Proteinuria_pos', 'Proteinuria (+)'),
        ('Hematuria_pos', 'Hematuria (+)'),
        ('TABACO', 'Tabaquismo'),
        ('OH', 'Consumo OH'),
        ('¿PPOO?', 'Pueblo originario'),
    ]
    rows = []
    for var, lbl in risk_factors:
        sub = df.dropna(subset=[var, 'CKD_eGFR_lt60'])
        if len(sub) < 10:
            continue
        ex = sub[sub[var]==1]; nx = sub[sub[var]==0]
        if len(ex)<5 or len(nx)<5:
            continue
        eck = (ex['CKD_eGFR_lt60']==1).sum(); nck = (nx['CKD_eGFR_lt60']==1).sum()
        if eck==0 or nck==0:
            rows.append({'Factor': lbl, 'RR': np.nan, 'lo': np.nan, 'hi': np.nan,
                         'p': np.nan, 'prev_ex': eck/len(ex)*100,
                         'prev_nx': nck/len(nx)*100, 'n_ex': len(ex), 'n_nx': len(nx)})
            continue
        rr = (eck/len(ex)) / (nck/len(nx))
        se = np.sqrt(1/eck - 1/len(ex) + 1/nck - 1/len(nx))
        lo = np.exp(np.log(rr) - 1.96*se)
        hi = np.exp(np.log(rr) + 1.96*se)
        p = chi2_pvalue(sub, var, 'CKD_eGFR_lt60')
        rows.append({'Factor': lbl, 'RR': rr, 'lo': lo, 'hi': hi, 'p': p,
                     'prev_ex': eck/len(ex)*100, 'prev_nx': nck/len(nx)*100,
                     'n_ex': len(ex), 'n_nx': len(nx)})

    if rows:
        rf_df = pd.DataFrame(rows).dropna(subset=['RR']).sort_values('RR')

        fig = go.Figure()
        for _, r in rf_df.iterrows():
            color = C_ACC if r['lo'] > 1 else (C_OK if r['hi'] < 1 else C_GRAY)
            fig.add_trace(go.Scatter(
                x=[r['lo'], r['hi']], y=[r['Factor'], r['Factor']],
                mode='lines', line=dict(color=color, width=2),
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=[r['RR']], y=[r['Factor']], mode='markers',
                marker=dict(color=color, size=12,
                            line=dict(color='white', width=1.5)),
                showlegend=False,
                hovertemplate=f"<b>{r['Factor']}</b><br>RR={r['RR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})<br>p={r['p']:.4f}<extra></extra>"
            ))
        fig.add_vline(x=1, line_dash='dash', line_color='gray')
        fig.update_layout(
            xaxis=dict(title='Riesgo Relativo (escala log)', type='log'),
            yaxis=dict(title=''),
            height=500, margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        rf_df_show = rf_df[['Factor','prev_ex','prev_nx','RR','lo','hi','p']].round(3)
        rf_df_show.columns = ['Factor','%ERC expuestos','%ERC no expuestos','RR','IC inf','IC sup','p']
        st.dataframe(rf_df_show, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Regresión logística multivariada")

    dat = df[['CKD_eGFR_lt60','Age','Sex','¿PPOO?','Obesity',
              'LE HAN DICHO QUE TIENE HTA','LE HAN DICHO QUE TIENE DM?',
              'ANTEC. FLIAR ERC','Proteinuria_pos']].copy()
    dat.columns = ['ckd','age','sex','ppoo','obese','hta_sr','dm_sr','famhx','prot']
    dat = dat.dropna()
    dat['ckd'] = dat['ckd'].fillna(0).astype(int)

    if len(dat) < 50 or dat['ckd'].sum() < 10:
        st.warning(f"Muestra insuficiente para regresión (N={len(dat)}, eventos={dat['ckd'].sum() if len(dat) else 0}).")
    else:
        try:
            model = smf.logit('ckd ~ age + sex + ppoo + obese + hta_sr + dm_sr + famhx + prot', data=dat).fit(disp=0)
            or_tab = pd.DataFrame({
                'OR': np.exp(model.params),
                'IC inf': np.exp(model.conf_int()[0]),
                'IC sup': np.exp(model.conf_int()[1]),
                'p': model.pvalues
            }).drop('Intercept')

            labels_map = {
                'age':'Edad (por año)','sex':'Sexo masculino','ppoo':'Pueblo originario',
                'obese':'Obesidad','hta_sr':'HTA autoreporte','dm_sr':'DM autoreporte',
                'famhx':'Antec familiar ERC','prot':'Proteinuria (+)'
            }
            or_tab.index = or_tab.index.map(lambda x: labels_map.get(x, x))

            fig = go.Figure()
            for var in or_tab.index:
                row = or_tab.loc[var]
                color = C_ACC if row['IC inf'] > 1 else (C_OK if row['IC sup'] < 1 else C_GRAY)
                fig.add_trace(go.Scatter(
                    x=[row['IC inf'], row['IC sup']], y=[var, var],
                    mode='lines', line=dict(color=color, width=2),
                    showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=[row['OR']], y=[var], mode='markers',
                    marker=dict(color=color, size=12, line=dict(color='white', width=1.5)),
                    showlegend=False,
                    hovertemplate=f"<b>{var}</b><br>OR={row['OR']:.2f} ({row['IC inf']:.2f}–{row['IC sup']:.2f})<br>p={row['p']:.4f}<extra></extra>"
                ))
            fig.add_vline(x=1, line_dash='dash', line_color='gray')
            fig.update_layout(
                xaxis=dict(title='Odds Ratio ajustado (escala log)', type='log'),
                yaxis_title='', height=400, margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"N={len(dat)} | Pseudo-R² McFadden = {model.prsquared:.3f} | AIC = {model.aic:.1f}")

            or_show = or_tab.round(3).reset_index().rename(columns={'index':'Variable'})
            st.dataframe(or_show, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"No se pudo ajustar el modelo: {e}")


# ═══════════════════════════════════════════════════════════
# TAB 6 — SCORE DE RIESGO
# ═══════════════════════════════════════════════════════════
with tab_score:
    st.header("Score de riesgo para tamizaje")

    st.markdown("""
    **Score propuesto (rango 0–13 puntos)**

    | Variable | Puntos |
    |---|---|
    | Edad <45 / 45–59 / 60–74 / ≥75 | 0 / 2 / 4 / 6 |
    | Sexo masculino | 1 |
    | Obesidad (IMC ≥30) | 1 |
    | HTA conocida (autoreporte) | 1 |
    | Antecedente familiar de ERC | 1 |
    | Proteinuria (+) | 2 |
    """)

    def age_pts(a):
        if pd.isna(a): return np.nan
        if a < 45: return 0
        elif a < 60: return 2
        elif a < 75: return 4
        else: return 6

    sc_df = df.copy()
    sc_df['s_age'] = sc_df['Age'].apply(age_pts)
    sc_df['s_sex'] = (sc_df['Sex'].fillna(0)==1).astype(int)
    sc_df['s_ob']  = (sc_df['Obesity'].fillna(0)==1).astype(int)
    sc_df['s_hta'] = (sc_df['LE HAN DICHO QUE TIENE HTA'].fillna(0)==1).astype(int)
    sc_df['s_fam'] = (sc_df['ANTEC. FLIAR ERC'].fillna(0)==1).astype(int)
    sc_df['s_prot']= (sc_df['Proteinuria_pos'].fillna(0)==1).astype(int)*2

    sc_df['Score'] = (sc_df[['s_age','s_sex','s_ob','s_hta','s_fam','s_prot']].sum(axis=1))
    sc_df.loc[sc_df['Age'].isna(), 'Score'] = np.nan

    valid = sc_df.dropna(subset=['Score','CKD_eGFR_lt60'])
    if len(valid) < 50:
        st.warning(f"Muestra insuficiente para el score (N válidos={len(valid)}).")
    else:
        valid['ckd'] = valid['CKD_eGFR_lt60'].fillna(0).astype(int)

        c1, c2 = st.columns(2)
        with c1:
            grp = valid.groupby('Score').agg(n=('ckd','size'), n_ckd=('ckd','sum'))
            grp['prev'] = grp['n_ckd']/grp['n']*100

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=grp.index, y=grp['n'], name='N pacientes',
                marker_color=C_LIGHT, opacity=0.7
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=grp.index, y=grp['prev'],
                name='Prevalencia ERC', mode='lines+markers+text',
                line=dict(color=C_ACC, width=3),
                marker=dict(size=10),
                text=[f'{p:.0f}%' for p in grp['prev']], textposition='top center',
            ), secondary_y=True)
            fig.update_xaxes(title_text='Score (puntos)')
            fig.update_yaxes(title_text='N pacientes', secondary_y=False)
            fig.update_yaxes(title_text='Prevalencia ERC (%)', secondary_y=True, range=[0,110])
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=20),
                              legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            try:
                auc = roc_auc_score(valid['ckd'], valid['Score'])
                fpr, tpr, thr = roc_curve(valid['ckd'], valid['Score'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    line=dict(color=C_MAIN, width=2.5),
                    name=f'Score (AUC={auc:.3f})'
                ))
                fig.add_trace(go.Scatter(
                    x=[0,1], y=[0,1], mode='lines',
                    line=dict(color=C_GRAY, dash='dash'),
                    name='No discriminación', showlegend=True
                ))
                fig.update_layout(
                    xaxis_title='1 - Especificidad',
                    yaxis_title='Sensibilidad',
                    height=400, margin=dict(l=10, r=10, t=30, b=20),
                    legend=dict(x=0.55, y=0.1)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"No se pudo calcular AUC: {e}")

        st.markdown("### Características operativas por punto de corte")
        rows = []
        for cut in sorted(valid['Score'].unique()):
            pred = (valid['Score'] >= cut).astype(int)
            tp = ((pred==1) & (valid['ckd']==1)).sum()
            fn = ((pred==0) & (valid['ckd']==1)).sum()
            fp = ((pred==1) & (valid['ckd']==0)).sum()
            tn = ((pred==0) & (valid['ckd']==0)).sum()
            sens = tp/(tp+fn) if (tp+fn) else 0
            esp  = tn/(tn+fp) if (tn+fp) else 0
            vpp  = tp/(tp+fp) if (tp+fp) else 0
            vpn  = tn/(tn+fn) if (tn+fn) else 0
            nnd  = 1/vpp if vpp>0 else np.inf
            rows.append({
                'Cutoff': f'≥{int(cut)}',
                'N tamizado': tp+fp,
                'VP': tp,'FN': fn,'FP': fp,'VN': tn,
                'Sensibilidad': round(sens,2),
                'Especificidad': round(esp,2),
                'VPP': round(vpp,2),
                'VPN': round(vpn,2),
                'NND': round(nnd,1) if not np.isinf(nnd) else '—'
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### 🧮 Calculadora individual")
        col1, col2, col3 = st.columns(3)
        with col1:
            i_age = st.slider("Edad", 18, 100, 55)
            i_sex = st.radio("Sexo", ['Mujer','Hombre'], horizontal=True)
        with col2:
            i_ob = st.checkbox("Obesidad (IMC ≥30)")
            i_hta = st.checkbox("HTA conocida")
        with col3:
            i_fam = st.checkbox("Antec familiar ERC")
            i_prot = st.checkbox("Proteinuria (+)")

        score_user = (
            age_pts(i_age) +
            (1 if i_sex=='Hombre' else 0) +
            (1 if i_ob else 0) +
            (1 if i_hta else 0) +
            (1 if i_fam else 0) +
            (2 if i_prot else 0)
        )
        # Buscar prevalencia en el bin más cercano
        bin_prev = valid[valid['Score']==score_user]
        if len(bin_prev) > 0:
            prev_at_score = (bin_prev['ckd']==1).sum()/len(bin_prev)*100
        else:
            prev_at_score = np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Score total", score_user)
        c2.metric("Prevalencia ERC en este score",
                  f"{prev_at_score:.1f}%" if pd.notna(prev_at_score) else "—",
                  help=f"Calculada sobre {len(bin_prev)} pacientes con score={score_user}")
        nivel = ('Bajo' if score_user<=2 else
                 'Intermedio' if score_user<=5 else
                 'Alto' if score_user<=7 else 'Muy alto')
        c3.metric("Categoría de riesgo", nivel)


# ═══════════════════════════════════════════════════════════
# TAB 7 — CONCORDANCIA
# ═══════════════════════════════════════════════════════════
with tab_concord:
    st.header("Concordancia eGFR 1ra vs 2da toma")

    sub = df.dropna(subset=['eGFR_1st','eGFR_2nd']).copy()
    if len(sub) < 5:
        st.warning(f"Solo {len(sub)} pacientes con segunda toma — insuficiente para análisis.")
    else:
        sub['delta'] = sub['eGFR_2nd'] - sub['eGFR_1st']
        sub['mean']  = (sub['eGFR_1st']+sub['eGFR_2nd'])/2
        bias = sub['delta'].mean()
        sd = sub['delta'].std()
        loa_lo, loa_hi = bias-1.96*sd, bias+1.96*sd

        try:
            r_p, p_p = stats.pearsonr(sub['eGFR_1st'], sub['eGFR_2nd'])
        except Exception:
            r_p, p_p = (np.nan, np.nan)

        mx, my = sub['eGFR_1st'].mean(), sub['eGFR_2nd'].mean()
        sx, sy = sub['eGFR_1st'].std(ddof=0), sub['eGFR_2nd'].std(ddof=0)
        cov = ((sub['eGFR_1st']-mx)*(sub['eGFR_2nd']-my)).mean()
        denom = sx**2+sy**2+(mx-my)**2
        ccc = 2*cov/denom if denom > 0 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N con ambas tomas", len(sub))
        c2.metric("Bias (Δ eGFR)", f"{bias:+.2f}")
        c3.metric("Pearson r", f"{r_p:.3f}" if pd.notna(r_p) else "—")
        c4.metric("Lin's CCC", f"{ccc:.3f}" if pd.notna(ccc) else "—",
                  help="<0.90 = pobre")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Scatter + diagonal**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub['eGFR_1st'], y=sub['eGFR_2nd'],
                mode='markers',
                marker=dict(color=C_MAIN, size=8, line=dict(color='white', width=1)),
                hovertemplate='1ra: %{x:.0f}<br>2da: %{y:.0f}<extra></extra>'
            ))
            mn, mx_ = min(sub['eGFR_1st'].min(), sub['eGFR_2nd'].min()), max(sub['eGFR_1st'].max(), sub['eGFR_2nd'].max())
            fig.add_trace(go.Scatter(x=[mn,mx_], y=[mn,mx_], mode='lines',
                                     line=dict(color=C_GRAY, dash='dash'),
                                     name='Línea de identidad', showlegend=False))
            fig.add_hline(y=60, line_color=C_ACC, line_dash='dot')
            fig.add_vline(x=60, line_color=C_ACC, line_dash='dot')
            fig.update_layout(
                xaxis_title='eGFR 1ra toma', yaxis_title='eGFR 2da toma',
                height=400, margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Bland-Altman**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub['mean'], y=sub['delta'], mode='markers',
                marker=dict(color=C_MAIN, size=8, line=dict(color='white', width=1)),
                hovertemplate='Media: %{x:.0f}<br>Δ: %{y:.1f}<extra></extra>'
            ))
            fig.add_hline(y=bias, line_color='black',
                          annotation_text=f'bias={bias:.1f}')
            fig.add_hline(y=loa_hi, line_color=C_ACC, line_dash='dash',
                          annotation_text=f'+1.96 DE = {loa_hi:.1f}')
            fig.add_hline(y=loa_lo, line_color=C_ACC, line_dash='dash',
                          annotation_text=f'-1.96 DE = {loa_lo:.1f}')
            fig.update_layout(
                xaxis_title='Promedio (1ra+2da)/2',
                yaxis_title='Δ eGFR (2da - 1ra)',
                height=400, margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        sub['ckd_1'] = (sub['eGFR_1st']<60).fillna(False).astype(int)
        sub['ckd_2'] = (sub['eGFR_2nd']<60).fillna(False).astype(int)
        tab = pd.crosstab(
            sub['ckd_1'].map({0:'≥60 (1ra)',1:'<60 (1ra)'}),
            sub['ckd_2'].map({0:'≥60 (2da)',1:'<60 (2da)'}),
            margins=True
        )
        st.markdown("### Reclasificación binaria (eGFR < 60)")
        st.dataframe(tab, use_container_width=True)

        try:
            n00 = ((sub['ckd_1']==0)&(sub['ckd_2']==0)).sum()
            n11 = ((sub['ckd_1']==1)&(sub['ckd_2']==1)).sum()
            n01 = ((sub['ckd_1']==0)&(sub['ckd_2']==1)).sum()
            n10 = ((sub['ckd_1']==1)&(sub['ckd_2']==0)).sum()
            N = n00+n01+n10+n11
            po = (n00+n11)/N
            pe = ((n00+n01)*(n00+n10) + (n10+n11)*(n01+n11)) / N**2
            kappa = (po-pe)/(1-pe) if (1-pe) != 0 else np.nan
            st.caption(f"**Kappa de Cohen = {kappa:.3f}** "
                       f"(<0.20 pobre, 0.21–0.40 leve, 0.41–0.60 moderada, 0.61–0.80 sustancial, >0.80 casi perfecta)")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# TAB 8 — DATOS
# ═══════════════════════════════════════════════════════════
with tab_data:
    st.header("Datos filtrados")
    st.caption(f"N actualmente seleccionado: **{len(df):,}**")

    cols_show = st.multiselect(
        "Columnas a mostrar",
        options=df.columns.tolist(),
        default=['ID','Age','Sex_lbl','Community','PPOO_lbl','BMI',
                 'BP_Final','eGFR_1st','CKD_KDIGO_G','Albuminuria_cat',
                 'KDIGO_risk']
    )
    if cols_show:
        st.dataframe(df[cols_show], use_container_width=True, height=500)

        csv = df[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar selección (CSV)",
            data=csv,
            file_name='ckd_filtered.csv',
            mime='text/csv'
        )

st.markdown("---")
st.caption(
    "Estudio de prevalencia de ERC · Región de la Araucanía. "
    "Las prevalencias aplican a la muestra de screening, no a la población general."
)
