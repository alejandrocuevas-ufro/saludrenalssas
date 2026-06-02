"""
DASHBOARD INTERACTIVO — Estudio de prevalencia de ERC
Región de la Araucanía, Chile  |  v4.2.14 — CKD-EPI 2021 | Planillas 01-05

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
from scipy.stats import pearsonr, spearmanr, linregress
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
import unicodedata
from pathlib import Path
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ERC Araucanía — Dashboard v4",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta KDIGO
C_MAIN  = '#1F3864'
C_TEAL  = '#2E7D9F'
C_ACC   = '#C0392B'
C_OK    = '#4DAF4A'
C_LIGHT = '#9FC2BA'
C_GRAY  = '#666666'
C_LGRAY = '#D5DDE8'
C_PPOO  = '#8B4513'
C_NOPP  = '#2E7D9F'
C_RURAL = '#5D4037'
C_URBAN = '#1565C0'

# Estadios KDIGO → color
KDIGO_COLORS = {
    'G1': '#4DAF4A', 'G2': '#88C640', 'G3a': '#FFDD00',
    'G3b': '#F28F00', 'G4': '#E83800', 'G5': '#9E0000',
}
RISK_COLORS = {
    'Bajo': '#A8D5A0', 'Moderado': '#FFDD00',
    'Alto': '#F08C5A', 'Muy alto': '#C53030',
}


def blend_with_white(hex_color, intensity):
    """Mezcla blanco con un color base. intensity 0=blanco, 1=color pleno."""
    intensity = max(0, min(1, float(intensity))) if pd.notna(intensity) else 0
    h = hex_color.lstrip('#')
    r,g,b = tuple(int(h[i:i+2], 16) for i in (0,2,4))
    r = int(255*(1-intensity) + r*intensity)
    g = int(255*(1-intensity) + g*intensity)
    b = int(255*(1-intensity) + b*intensity)
    return f'#{r:02X}{g:02X}{b:02X}'

st.markdown("""
<style>
    .stMetric { background:#f8f9fa; padding:10px; border-radius:6px;
                border-left:4px solid #1F3864; }
    h1 { color:#1F3864; }
    h2 { color:#1F3864; border-bottom:2px solid #9FC2BA; padding-bottom:4px; }
    h3 { color:#2E7D9F; }
    .stTabs [data-baseweb="tab-list"] { gap:6px; }
    .stTabs [data-baseweb="tab"] { background:#f0f2f5; padding:8px 16px; border-radius:4px; }
    .stTabs [aria-selected="true"] { background:#1F3864 !important; color:white !important; }
    .kdigo-badge { display:inline-block; padding:3px 10px; border-radius:12px;
                   font-weight:bold; font-size:0.85em; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# RURALIDAD
# ═══════════════════════════════════════════════════════════
# Valores armonizados con la adenda corregida del informe.
# Se usa una versión canónica para mostrar nombres propios con tildes,
# y una clave normalizada para hacer matching robusto contra la planilla.
RURALIDAD_CANONICA = {
    'Freire':48.0,'Melipeuco':72.3,'Cholchol':71.8,'Gorbea':42.1,
    'Curarrehue':78.9,'Cunco':60.4,'Temuco':6.2,'Pitrufquén':38.7,
    'Pucón':37.0,'Villarrica':34.7,'Padre Las Casas':11.3,
    'Galvarino':63.2,'Vilcún':66.2,'Lautaro':26.8,
    'Teodoro Schmidt':64.9,'Loncoche':45.5,'Curacautín':28.3,
    'Puerto Saavedra':63.7,'Nueva Imperial':48.2,
    'La Pintana':0.0,'Estación Central':0.0,'La Granja':0.0,
}

def norm_key(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = ' '.join(s.split()).lower()
    return s

RURALIDAD_NORM = {norm_key(k): v for k, v in RURALIDAD_CANONICA.items()}
COMUNA_CANONICA_NORM = {norm_key(k): k for k in RURALIDAD_CANONICA.keys()}
# Alias explícitos para variantes frecuentes de codificación/capitalización.
COMUNA_ALIAS = {
    'estacion central': 'Estación Central',
    'nueva imperial': 'Nueva Imperial',
    'padre las casas': 'Padre Las Casas',
    'puerto saavedra': 'Puerto Saavedra',
    'teodoro schmidt': 'Teodoro Schmidt',
}
COMUNA_CANONICA_NORM.update({norm_key(k): v for k, v in COMUNA_ALIAS.items()})

# ═══════════════════════════════════════════════════════════
# HELPERS ESTADÍSTICOS
# ═══════════════════════════════════════════════════════════
def wilson(k, n):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k/n; z = 1.96; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (p*100, (c-h)*100, (c+h)*100)

def format_p(p):
    if pd.isna(p): return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def _holm_adjust(pvals):
    """Holm-Bonferroni adjustment without extra dependencies."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)
    ok = np.where(~np.isnan(pvals))[0]
    if len(ok) == 0: return out
    order = ok[np.argsort(pvals[ok])]
    m = len(order); running = 0
    for rank, idx in enumerate(order):
        adj = min((m-rank) * pvals[idx], 1.0)
        running = max(running, adj)
        out[idx] = running
    return out

def smart_categorical_test(df, group, outcome):
    """Test global para outcome categórico/binario vs grupos.
    Usa Fisher en 2x2 con frecuencias esperadas bajas; si no, Chi².
    """
    sub = df.dropna(subset=[group, outcome]).copy()
    if len(sub) < 5 or sub[group].nunique() < 2 or sub[outcome].nunique() < 2:
        return {'method':'No aplicable', 'p':np.nan, 'note':'datos insuficientes'}
    tab = pd.crosstab(sub[group], sub[outcome])
    try:
        chi2, p_chi, _, expected = stats.chi2_contingency(tab)
        low_exp = (expected < 5).sum()
        low_prop = low_exp / expected.size
        if tab.shape == (2,2) and low_exp > 0:
            _, p = stats.fisher_exact(tab)
            return {'method':'Fisher exacto 2×2', 'p':p, 'note':'usado por frecuencias esperadas <5'}
        note = ''
        if low_prop > 0.20:
            note = 'precaución: >20% de celdas con frecuencia esperada <5'
        return {'method':'Chi² de independencia', 'p':p_chi, 'note':note}
    except Exception as e:
        return {'method':'No calculable', 'p':np.nan, 'note':str(e)}

def pairwise_categorical_posthoc(df, group, outcome, min_n=5):
    sub = df.dropna(subset=[group, outcome]).copy()
    levels = [x for x in sub[group].dropna().unique()]
    rows=[]
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            a,b = levels[i], levels[j]
            tmp = sub[sub[group].isin([a,b])]
            if tmp[group].value_counts().min() < min_n: continue
            tab = pd.crosstab(tmp[group], tmp[outcome])
            if tab.shape != (2,2): continue
            try:
                chi2, p_chi, _, exp = stats.chi2_contingency(tab)
                method = 'Fisher' if (exp < 5).any() else 'Chi² 2×2'
                p = stats.fisher_exact(tab)[1] if method == 'Fisher' else p_chi
                pa = tmp.loc[tmp[group]==a, outcome].mean()*100
                pb = tmp.loc[tmp[group]==b, outcome].mean()*100
                rows.append({'Comparación':f'{a} vs {b}', 'Δ positividad pp':pa-pb,
                             'p':p, 'Método':method})
            except Exception:
                pass
    if not rows: return pd.DataFrame()
    out = pd.DataFrame(rows)
    out['p ajustado Holm'] = _holm_adjust(out['p'].values)
    out['Significativo'] = out['p ajustado Holm'] < 0.05
    return out.sort_values('p ajustado Holm')

def smart_continuous_test(df, group, value):
    """Test global para variable continua según grupos: t/Mann-Whitney o ANOVA/Kruskal."""
    sub = df.dropna(subset=[group, value]).copy()
    groups = [g[value].astype(float).values for _, g in sub.groupby(group) if len(g) >= 3]
    labels = [k for k, g in sub.groupby(group) if len(g) >= 3]
    if len(groups) < 2:
        return {'method':'No aplicable', 'p':np.nan, 'note':'datos insuficientes', 'posthoc':pd.DataFrame()}
    normal = True
    for arr in groups:
        if len(arr) < 3:
            normal = False; break
        # Shapiro se limita a 5000 obs por restricción del test
        sample = arr if len(arr) <= 5000 else np.random.default_rng(123).choice(arr, 5000, replace=False)
        try:
            if stats.shapiro(sample).pvalue < 0.05:
                normal = False; break
        except Exception:
            normal = False; break
    try:
        lev_p = stats.levene(*groups, center='median').pvalue
    except Exception:
        lev_p = np.nan
    equal_var = pd.notna(lev_p) and lev_p >= 0.05
    if len(groups) == 2:
        if normal:
            res = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var, nan_policy='omit')
            method = 't de Student' if equal_var else 't de Welch'
            note = 'normalidad compatible; varianzas homogéneas' if equal_var else 'normalidad compatible; varianzas desiguales'
        else:
            res = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            method = 'Mann–Whitney U'
            note = 'distribución no normal o asimétrica'
        return {'method':method, 'p':res.pvalue, 'note':note, 'posthoc':pd.DataFrame()}
    else:
        if normal and equal_var:
            res = stats.f_oneway(*groups)
            method = 'ANOVA una vía'
            note = 'normalidad y homocedasticidad compatibles'
            pair_method = 't pareada independiente'
        else:
            res = stats.kruskal(*groups)
            method = 'Kruskal–Wallis'
            note = 'no normalidad y/o varianzas heterogéneas'
            pair_method = 'Mann–Whitney'
        rows=[]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                if pair_method.startswith('t'):
                    p = stats.ttest_ind(groups[i], groups[j], equal_var=False, nan_policy='omit').pvalue
                else:
                    p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided').pvalue
                rows.append({'Comparación':f'{labels[i]} vs {labels[j]}', 'p':p, 'Método':pair_method})
        ph = pd.DataFrame(rows)
        if len(ph):
            ph['p ajustado Holm'] = _holm_adjust(ph['p'].values)
            ph['Significativo'] = ph['p ajustado Holm'] < 0.05
            ph = ph.sort_values('p ajustado Holm')
        return {'method':method, 'p':res.pvalue, 'note':note, 'posthoc':ph}

def style_sig(df, pcols=('p','p ajustado Holm')):
    """Resalta en pastel filas/celdas significativas en tablas Streamlit."""
    pcols = [c for c in pcols if c in df.columns]
    if not pcols: return df
    def row_style(row):
        sig = False
        for c in pcols:
            try:
                if pd.notna(row[c]) and float(row[c]) < 0.05:
                    sig = True
            except Exception:
                pass
        return ['background-color: #EAF6EA' if sig else '' for _ in row]
    return df.style.apply(row_style, axis=1)


def humanize_varname(c):
    """Etiqueta legible para variables binarias estandarizadas."""
    txt = str(c)
    replacements = {
        'OtraCondicion_': 'Otra condición: ',
        'AF_': 'Antecedente familiar: ',
        'Dg_': 'Diagnóstico: ',
        'AINEs_': 'AINEs: ',
        'Eval_': 'Evaluación: ',
        'Accion_': 'Acción: ',
        'Acción_': 'Acción: ',
        'Pesquisa_': 'Pesquisa: ',
    }
    for k, v in replacements.items():
        txt = txt.replace(k, v)
    txt = txt.replace('_', ' ').replace('  ', ' ').strip()
    return txt

def is_binary_series(s):
    vals = pd.Series(s).dropna().unique()
    if len(vals) == 0:
        return False
    try:
        return set(pd.Series(vals).astype(float).unique()).issubset({0.0, 1.0})
    except Exception:
        return False


def _norm_varname_for_filter(c):
    """Normaliza nombres de variables para filtros semánticos robustos."""
    t = norm_key(c)
    if pd.isna(t):
        return ''
    return str(t).replace('_', ' ').replace('-', ' ')


def is_nonanalytic_indicator_name(c):
    """Identifica columnas auxiliares que no deben entrar en análisis de riesgo."""
    t = _norm_varname_for_filter(c)
    excluded_terms = [
        'revisar', 'revision', 'dato faltante', 'sin informacion',
        'sin alteraciones relevantes', 'antecedente no especificado',
        'sin antecedentes declarados', 'no especificado', 'diagnostico provisional obs',
    ]
    return any(term in t for term in excluded_terms)


def combine_binary_any(df, cols):
    """Combina variables 0/1 equivalentes: 1 si cualquiera es 1; 0 si hay algún 0 y ningún 1; NaN si todo falta."""
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return pd.Series(np.nan, index=df.index, dtype='float')
    mat = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in avail], axis=1)
    any_pos = (mat == 1).any(axis=1)
    any_zero = (mat == 0).any(axis=1)
    return pd.Series(np.where(any_pos, 1.0, np.where(any_zero, 0.0, np.nan)), index=df.index)


def standardized_binary_columns(df):
    """Columnas estandarizadas 01-05 con codificación 0/1 usable en RR/estratificación."""
    prefixes = ('OtraCondicion_', 'AF_', 'Dg_', 'AINEs_', 'Eval_', 'Accion_', 'Acción_', 'Pesquisa_')
    cols = []
    for c in df.columns:
        if is_nonanalytic_indicator_name(c):
            continue
        if c.startswith(prefixes) and is_binary_series(df[c]):
            # Excluir indicadores redundantes o poco interpretables si aparecen vacíos.
            if df[c].notna().sum() >= 10 and pd.to_numeric(df[c], errors='coerce').sum(skipna=True) >= 5:
                cols.append(c)
    return cols

def rr_rows_for_binary_vars(df, variables, outcome='CKD60', min_exposed=5, min_unexposed=5):
    rows=[]
    for var, lbl in variables:
        if var not in df.columns:
            continue
        sub=df.dropna(subset=[var,outcome]).copy()
        if len(sub)<10:
            continue
        try:
            sub[var]=pd.to_numeric(sub[var], errors='coerce')
        except Exception:
            continue
        sub=sub.dropna(subset=[var,outcome])
        ex=sub[sub[var]==1]; nx=sub[sub[var]==0]
        if len(ex)<min_exposed or len(nx)<min_unexposed:
            continue
        eck=int((ex[outcome]==1).sum()); nck=int((nx[outcome]==1).sum())
        if eck==0 or nck==0:
            continue
        rr=(eck/len(ex))/(nck/len(nx))
        se=np.sqrt(1/eck-1/len(ex)+1/nck-1/len(nx))
        lo=np.exp(np.log(rr)-1.96*se); hi=np.exp(np.log(rr)+1.96*se)
        test = smart_categorical_test(sub,var,outcome)
        rows.append({'Factor':lbl,'Variable':var,'N expuestos':len(ex), 'N no expuestos':len(nx),
                     'n ERC expuestos':eck, 'n ERC no exp.':nck,
                     '%ERC expuestos':eck/len(ex)*100,'%ERC no exp.':nck/len(nx)*100,
                     'RR':rr,'IC inf':lo,'IC sup':hi,'p':test.get('p',np.nan),'Test':test.get('method','')})
    return pd.DataFrame(rows)


def standardized_risk_columns(df, include_groups=('antecedentes','familiares','aines')):
    """Columnas 0/1 estandarizadas útiles como exposiciones basales/exploratorias.

    Se excluyen por diseño variables de evaluación, acciones o pesquisa clínica posterior
    porque no representan exposiciones basales para interpretar como riesgo.
    """
    prefix_map = {
        'antecedentes': ('OtraCondicion_', 'Dg_'),
        'familiares': ('AF_',),
        'aines': ('AINEs_',),
    }
    prefixes = tuple(p for g in include_groups for p in prefix_map.get(g, ()))
    cols = []
    # Duplicados/no exposición que no conviene presentar como factores de riesgo.
    # AINEs_diario se deriva de la base principal; si existe Planilla 03 se usa AINEs_Uso_Diario.
    aines_exclude = {'AINEs_diario', 'AINEs_diario_modelo', 'AINEs_NoUso_Reportado'}
    for c in df.columns:
        if is_nonanalytic_indicator_name(c):
            continue
        if c in aines_exclude:
            continue
        if c.startswith(prefixes) and is_binary_series(df[c]):
            if df[c].notna().sum() >= 10 and pd.to_numeric(df[c], errors='coerce').sum(skipna=True) >= 5:
                cols.append(c)
    return cols


def prevalence_ratio_row(df, var, lbl, outcome='CKD60', covariates=None,
                         min_exposed=5, min_unexposed=5, min_events=1):
    """Calcula RP/RR cruda o ajustada para una exposición binaria.

    - Crudo: razón de prevalencias directa con IC log aproximado.
    - Ajustado: GLM Poisson con enlace log y varianza robusta HC3.
    """
    if var not in df.columns or outcome not in df.columns:
        return None
    covariates = [c for c in (covariates or []) if c in df.columns and c != var]
    need = [var, outcome] + covariates
    sub = df[need].copy()
    for c in need:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sub = sub.dropna(subset=need)
    if len(sub) < 10:
        return None
    sub = sub[sub[var].isin([0,1]) & sub[outcome].isin([0,1])]
    if len(sub) < 10:
        return None
    ex = sub[sub[var] == 1]
    nx = sub[sub[var] == 0]
    if len(ex) < min_exposed or len(nx) < min_unexposed:
        return None
    eck = int((ex[outcome] == 1).sum())
    nck = int((nx[outcome] == 1).sum())
    if eck < min_events or nck < min_events:
        return None

    base = {
        'Factor': lbl, 'Variable': var,
        'N modelo': len(sub), 'N expuestos': len(ex), 'N no expuestos': len(nx),
        'n ERC expuestos': eck, 'n ERC no exp.': nck,
        '%ERC expuestos': eck/len(ex)*100, '%ERC no exp.': nck/len(nx)*100,
    }

    if not covariates:
        rr = (eck/len(ex))/(nck/len(nx))
        se = np.sqrt(1/eck - 1/len(ex) + 1/nck - 1/len(nx))
        lo = np.exp(np.log(rr) - 1.96*se)
        hi = np.exp(np.log(rr) + 1.96*se)
        test = smart_categorical_test(sub, var, outcome)
        base.update({'RP': rr, 'IC inf': lo, 'IC sup': hi, 'p': test.get('p', np.nan),
                     'Método': 'Crudo', 'Ajuste': 'Sin ajuste', 'Test': test.get('method', '')})
        return base

    try:
        y = sub[outcome].astype(float)
        X = sub[[var] + covariates].astype(float).rename(columns={var: 'EXPOSURE'})
        X = sm.add_constant(X, has_constant='add')
        mod = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type='HC3')
        beta = mod.params['EXPOSURE']
        se = mod.bse['EXPOSURE']
        base.update({'RP': float(np.exp(beta)),
                     'IC inf': float(np.exp(beta - 1.96*se)),
                     'IC sup': float(np.exp(beta + 1.96*se)),
                     'p': float(mod.pvalues['EXPOSURE']),
                     'Método': 'Poisson robusto',
                     'Ajuste': ', '.join(covariates),
                     'Test': 'GLM Poisson robusto'})
        return base
    except Exception as e:
        base.update({'RP': np.nan, 'IC inf': np.nan, 'IC sup': np.nan, 'p': np.nan,
                     'Método': 'No calculable', 'Ajuste': ', '.join(covariates), 'Test': str(e)})
        return base


def prevalence_ratio_table(df, variables, outcome='CKD60', covariates=None):
    rows = []
    for var, lbl in variables:
        r = prevalence_ratio_row(df, var, lbl, outcome=outcome, covariates=covariates)
        if r is not None:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.dropna(subset=['RP'])
    return out


def add_rural_binary(df):
    if 'Rural_bin' not in df.columns and 'Rural_lbl' in df.columns:
        df['Rural_bin'] = np.where(df['Rural_lbl'].eq('Rural (≥30%)'), 1.0,
                                   np.where(df['Rural_lbl'].eq('Urbano (<30%)'), 0.0, np.nan))
    return df


def plot_pr_forest(pr_df, title_x='Razón de positividad (escala log)'):
    fig = go.Figure()
    if pr_df is None or len(pr_df) == 0:
        return fig
    plot_df = pr_df.sort_values('RP').copy()
    for _, r in plot_df.iterrows():
        col = (C_ACC if r['IC inf'] > 1 else C_OK if r['IC sup'] < 1 else C_GRAY)
        fig.add_trace(go.Scatter(x=[r['IC inf'], r['IC sup']], y=[r['Factor'], r['Factor']],
                                 mode='lines', line=dict(color=col, width=2),
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[r['RP']], y=[r['Factor']], mode='markers',
                                 marker=dict(color=col, size=12, line=dict(color='white', width=1.5)),
                                 showlegend=False,
                                 hovertemplate=(f"<b>{r['Factor']}</b><br>RP={r['RP']:.2f} "
                                                f"({r['IC inf']:.2f}–{r['IC sup']:.2f})"
                                                f"<br>p={r['p']:.4f}<br>N modelo={int(r['N modelo'])}"
                                                f"<extra></extra>")))
    fig.add_vline(x=1, line_dash='dash', line_color='gray')
    fig.update_layout(xaxis=dict(title=title_x, type='log'),
                      height=max(460, 28*len(plot_df)),
                      margin=dict(l=10, r=10, t=20, b=20))
    return fig

# compatibilidad con código antiguo: ahora delega en la selección automática categórica
def chi2p(df, var, out):
    return smart_categorical_test(df, var, out).get('p', np.nan)

def bar_annotations(x_vals, y_vals, hi_vals, labels, is_horizontal=False, font_size=11):
    """Return a list of Plotly annotations placed above the upper error bar whisker."""
    anns = []
    for x, y, hi, lbl in zip(x_vals, y_vals, hi_vals, labels):
        if pd.isna(y) or pd.isna(hi): continue
        offset = max(hi * 0.06, 1.0)
        if is_horizontal:
            anns.append(dict(x=hi+offset, y=x, text=lbl, showarrow=False,
                xanchor='left', yanchor='middle',
                font=dict(size=font_size, color='#333')))
        else:
            anns.append(dict(x=x, y=hi+offset, text=lbl, showarrow=False,
                xanchor='center', yanchor='bottom',
                font=dict(size=font_size, color='#333')))
    return anns

def yn(s):
    if pd.isna(s): return np.nan
    t = str(s).strip().lower()
    if t in ('sí','si','yes','1','1.0','true'): return 1
    if t in ('no','0','0.0','false'): return 0
    return np.nan

def kdigo_g(e):
    if pd.isna(e): return np.nan
    if e >= 90: return 'G1'
    if e >= 60: return 'G2'
    if e >= 45: return 'G3a'
    if e >= 30: return 'G3b'
    if e >= 15: return 'G4'
    return 'G5'

def kdigo_risk(g, a):
    m = {
        ('G1','A1'):'Bajo',('G1','A2'):'Moderado',('G1','A3'):'Alto',
        ('G2','A1'):'Bajo',('G2','A2'):'Moderado',('G2','A3'):'Alto',
        ('G3a','A1'):'Moderado',('G3a','A2'):'Alto',('G3a','A3'):'Muy alto',
        ('G3b','A1'):'Alto',('G3b','A2'):'Muy alto',('G3b','A3'):'Muy alto',
        ('G4','A1'):'Muy alto',('G4','A2'):'Muy alto',('G4','A3'):'Muy alto',
        ('G5','A1'):'Muy alto',('G5','A2'):'Muy alto',('G5','A3'):'Muy alto',
    }
    if pd.isna(g) or pd.isna(a): return np.nan
    return m.get((g, a), np.nan)

def alb_cat(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s in ('neg','negative','negativo','0'): return 'A1'
    try:
        f = float(s); return 'A2' if f <= 30 else 'A3'
    except: return np.nan

# ═══════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Cargando y preparando datos…")
def load_data(main_path, p01=None, p02=None, p03=None, p04=None, p05=None):
    df = pd.read_excel(main_path, sheet_name='Datos_analisis_curadomanual')

    # Excluir creatinina > 10
    df = df[df['Creatinine_1st'] <= 10].copy()

    # ── eGFR principal = CKD-EPI 2021 ──
    df['eGFR'] = df['eGFR_1st_CKD_EPI_2021_calc']
    df['eGFR_orig'] = df['eGFR_1st']

    # ── Sexo (texto en v3) ──
    df['Sex_lbl'] = df['Sex'].astype(str).str.strip()
    df['Sex_M']   = (df['Sex_lbl'] == 'Hombre').astype(int)

    # ── eGFR comparativo = MDRD-4 clásico usado por Zúñiga 2011 ──
    # Fórmula reportada: 186 × creatinina^-1.154 × edad^-0.203 × 0.742 si mujer
    # Se mantiene solo como variable auxiliar para comparación en Literatura,
    # sin modificar la definición principal CKD-EPI 2021 del dashboard.
    df['eGFR_MDRD4_Zuniga'] = np.where(
        df['Creatinine_1st'].notna() & df['Age'].notna() & (df['Creatinine_1st'] > 0) & (df['Age'] > 0),
        186 * (df['Creatinine_1st'] ** -1.154) * (df['Age'] ** -0.203)
        * np.where(df['Sex_lbl'].eq('Mujer'), 0.742, 1.0),
        np.nan
    )


    # ── PPOO ──
    df['PPOO_n']   = df['¿PPOO?'].apply(yn)
    df['PPOO_lbl'] = df['PPOO_n'].map({1: 'PPOO', 0: 'No PPOO'})

    # ── Edad ──
    df['Age_grp'] = pd.cut(df['Age'], [-1,29.999,44.999,59.999,74.999,200],
                           labels=['<30','30-44','45-59','60-74','≥75'])

    # ── Autoreportes (Sí/No/texto) ──
    for src, dst in [
        ('LE HAN DICHO QUE TIENE HTA','HTA_sr'),
        ('LE HAN DICHO QUE TIENE DM?','DM_sr'),
        ('HISTORIA PERSONAL LITIASIS','Litiasis'),
        ('ITU RECURRENTE','ITU'),
        ('ANTEC. FAMILIAR ERC','FamHx'),
        ('TTO_ERC','TTO_ERC_b'),
        ('TABACO','Tabaco'),
        ('OH','OH_b'),
        ('Glucose_gt200','Glc200'),
    ]:
        if src in df.columns:
            df[dst] = df[src].apply(yn)

    # ── PA ──
    df['HTA_meas'] = df['BP_Final'].isin(
        ['Stage 1 HTN','Stage 2 HTN','Hypertensive Crisis']).astype(float)
    df.loc[df['BP_Final'].isin(['left blank']) | df['BP_Final'].isna(), 'HTA_meas'] = np.nan

    # ── Proteinuria / hematuria ──
    def dippos(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if s in ('neg','negative','negativo','0'): return 0
        try: return 1 if float(s) > 0 else 0
        except: return 1 if s not in ('','nan') else np.nan
    df['prot_pos']  = df['Proteinuria'].apply(dippos)
    df['blood_pos'] = df['Blood'].apply(dippos)
    df['Alb_cat']   = df['Proteinuria'].apply(alb_cat)

    # ── KDIGO ──
    df['KDIGO_G']    = df['eGFR'].apply(kdigo_g)
    df['KDIGO_risk'] = df.apply(lambda r: kdigo_risk(r['KDIGO_G'], r['Alb_cat']), axis=1)

    # ── ERC definiciones ──
    df['CKD60'] = np.where(df['eGFR'].notna(), (df['eGFR'] < 60).astype(float), np.nan)
    df['CKD_exp'] = np.where(
        df['eGFR'].notna() | df['prot_pos'].notna(),
        ((df['eGFR'] < 60) | (df['prot_pos'] == 1)).astype(float), np.nan)

    # ── BMI (usa BMI_Category si existe) ──
    if 'BMI_Category' in df.columns:
        df['BMI_cat'] = df['BMI_Category'].astype(str).str.strip()
        df['Obesity']  = df['BMI_cat'].str.contains('Obesidad', na=False).astype(float)
        df.loc[df['BMI_Category'].isna(), 'Obesity'] = np.nan
    else:
        df['BMI_cat'] = np.nan; df['Obesity'] = np.nan

    # ── AINEs ──
    if 'Consumo AINEs' in df.columns:
        aines_map = {
            'Nunca': 0, 'Never': 0,
            'Cuando sea necesario': 1,
            'Unas pocas al mes': 2,
            '1-4 al día': 3, '5-10 al día': 4,
        }
        df['AINEs_ord'] = df['Consumo AINEs'].map(aines_map)
        df['AINEs_diario'] = (df['AINEs_ord'] >= 3).astype(float)
        df.loc[df['AINEs_ord'].isna(), 'AINEs_diario'] = np.nan
    else:
        df['AINEs_ord'] = np.nan; df['AINEs_diario'] = np.nan

    # ── Comunas y ruralidad ──
    df['Community_key'] = df['Community'].apply(norm_key)
    df['Community_std'] = df['Community_key'].map(COMUNA_CANONICA_NORM)
    # Si aparece una comuna no incluida en el diccionario, se conserva como nombre propio aproximado.
    df['Community_std'] = df['Community_std'].fillna(
        df['Community'].astype(str).str.strip().str.title().replace({'Nan': np.nan})
    )
    df['Ruralidad_pct'] = df['Community_key'].map(RURALIDAD_NORM)
    df['Rural_lbl']     = df['Ruralidad_pct'].apply(
        lambda x: 'Rural (≥30%)' if pd.notna(x) and x >= 30 else
                  ('Urbano (<30%)' if pd.notna(x) else np.nan))

    # ── Merge planillas de texto libre ──
    # Planilla 01 — OtraCondicion
    if p01 is not None:
        try:
            d01 = pd.read_excel(p01, sheet_name='Planilla_estandarizada')
            c01 = ['ID'] + [c for c in d01.columns if c.startswith('OtraCondicion_')]
            df = df.merge(d01[c01], on='ID', how='left')
        except: pass

    # Planilla 02 — Antecedentes familiares
    if p02 is not None:
        try:
            d02 = pd.read_excel(p02, sheet_name='Estandarizado_AF')
            c02 = ['ID'] + [c for c in d02.columns if c.startswith('AF_')]
            d02a = d02.groupby('ID')[c02[1:]].max().reset_index()
            df = df.merge(d02a, on='ID', how='left')
        except: pass

    # Planilla 03 — AINEs
    if p03 is not None:
        try:
            d03 = pd.read_excel(p03, sheet_name='AINEs_estandarizado')
            c03 = ['ID','AINEs_Categoria_Modelo','AINEs_Ordinal_Exposicion',
                   'AINEs_UsoCualquierFrecuencia','AINEs_NoUso_Reportado',
                   'AINEs_Uso_PRN_NoCuantificado','AINEs_Uso_OcasionalMensual',
                   'AINEs_Uso_Diario','AINEs_Uso_Diario_Alto_5a10']
            df = df.merge(d03[[c for c in c03 if c in d03.columns]], on='ID', how='left')
        except: pass

    # Planilla 04 — Diagnósticos
    if p04 is not None:
        try:
            d04 = pd.read_excel(p04, sheet_name='Variables_modelo')
            c04 = ['ID'] + [c for c in d04.columns if c.startswith('Dg_')]
            df = df.merge(d04[c04], on='ID', how='left')
        except: pass

    # Planilla 05 — Evaluaciones/Acciones
    if p05 is not None:
        try:
            d05 = pd.read_excel(p05, sheet_name='Para_modelo')
            df = df.merge(d05, on='ID', how='left')
        except: pass

    # ── Variables canónicas / compuestas para evitar duplicados entre fuentes ──
    # AINEs: si se carga la Planilla 03, se usa su codificación estandarizada;
    # si no está disponible, se mantiene la variable derivada desde la base principal como respaldo.
    if 'AINEs_Uso_Diario' in df.columns:
        df['AINEs_diario_modelo'] = combine_binary_any(df, ['AINEs_Uso_Diario'])
    else:
        df['AINEs_diario_modelo'] = df.get('AINEs_diario', pd.Series(np.nan, index=df.index))

    # Condiciones potencialmente equivalentes reportadas en distintas fuentes.
    # Se combinan para el bloque de factores clínicos predefinidos.
    df['Tiroides_global'] = combine_binary_any(df, ['Dg_Tiroides', 'OtraCondicion_Tiroides'])
    df['DLP_global'] = combine_binary_any(df, ['Dg_DLP_Hipertrigliceridemia', 'OtraCondicion_DLP'])
    df['Cardio_global'] = combine_binary_any(df, ['Dg_Cardiovascular_Global', 'OtraCondicion_Cardiovascular'])
    df['Litiasis_global'] = combine_binary_any(df, ['Litiasis', 'Dg_LitiasisRenal_Nefrocalcinosis'])
    df['ITU_global'] = combine_binary_any(df, ['ITU', 'Dg_ITU_Pielonefritis'])
    df['Tabaco_global'] = combine_binary_any(df, ['Tabaco', 'Dg_Tabaquismo'])
    df['OH_global'] = combine_binary_any(df, ['OH_b', 'Dg_Alcohol_Problema'])

    # FamHx combinado (formulario + planilla 02)
    fam_form = df.get('FamHx', pd.Series(np.nan, index=df.index))
    fam_erc  = df.get('AF_ERC_IRC', pd.Series(np.nan, index=df.index))
    fam_hd   = df.get('AF_Hemodialisis_Dialisis', pd.Series(np.nan, index=df.index))
    df['FamHx_ERC'] = np.where(
        (fam_form == 1) | (fam_erc == 1) | (fam_hd == 1), 1.0,
        np.where((fam_form == 0) & ((fam_erc == 0) | fam_erc.isna()) &
                 ((fam_hd == 0) | fam_hd.isna()), 0.0, np.nan))

    # Monorreno / asimetría renal: diagnóstico estructurado + texto estandarizado si lo menciona explícitamente.
    df['Monorreno'] = combine_binary_any(df, ['Dg_Monorreno_AsimetriaRenal'])
    if 'CUÁL OTRA CONDICION PERSONAL (Estandarizado)' in df.columns:
        df['Monorreno'] = np.where(
            df['CUÁL OTRA CONDICION PERSONAL (Estandarizado)'].str.contains(
                'monorreno|riñón único', case=False, na=False), 1.0, df['Monorreno'])

    # Cardiopatía, tiroides y DLP consolidadas desde diagnóstico + otra condición personal.
    df['Cardio']    = df['Cardio_global']
    df['Tiroides']  = df['Tiroides_global']
    df['DLP_dg']    = df['DLP_global']

    # ERC oculta
    df['ERC_oculta'] = np.where(
        (df['CKD60'] == 1) & df['Creatinine_1st'].notna(),
        (df['Creatinine_1st'] <= 1.0).astype(float), np.nan)

    # Score clínico (4 variables — Informe 2 / v4)
    def age_pts(a):
        if pd.isna(a): return np.nan
        if a < 45: return 0
        if a < 60: return 1
        if a < 75: return 2
        return 4
    df['sc_age']  = df['Age'].apply(age_pts)
    df['sc_prot'] = np.where(df['prot_pos'] == 1, 2, 0)
    df['sc_ob']   = np.where(df['Obesity'] == 1, 1, 0)
    df['sc_fam']  = np.where(df['FamHx_ERC'] == 1, 1, 0)
    df['SCORE']   = df['sc_age'] + df['sc_prot'] + df['sc_ob'] + df['sc_fam']
    df.loc[df['sc_age'].isna(), 'SCORE'] = np.nan

    return df


# ═══════════════════════════════════════════════════════════
# SIDEBAR — carga y filtros
# ═══════════════════════════════════════════════════════════
st.sidebar.title("🩺 ERC Araucanía v4")
st.sidebar.caption("CKD-EPI 2021 | Planillas 01-05 | Ruralidad comunal")

with st.sidebar.expander("📁 Archivos de datos", expanded=False):
    main_f = st.file_uploader("Base principal: ckd_data_v3_3.xlsx", type=['xlsx'], key='main')
    p01_f  = st.file_uploader("01_otra_condicion_salud.xlsx", type=['xlsx'], key='p01')
    p02_f  = st.file_uploader("02_antecedentes_familiares.xlsx", type=['xlsx'], key='p02')
    p03_f  = st.file_uploader("03_consumo_aines.xlsx", type=['xlsx'], key='p03')
    p04_f  = st.file_uploader("04_otros_diagnosticos.xlsx", type=['xlsx'], key='p04')
    p05_f  = st.file_uploader("05_evaluacion_medica.xlsx", type=['xlsx'], key='p05')

@st.cache_data
def _load(mf, p1, p2, p3, p4, p5):
    return load_data(mf, p1, p2, p3, p4, p5)

if main_f is not None:
    df_raw = _load(main_f, p01_f, p02_f, p03_f, p04_f, p05_f)
else:
    def _first_existing(patterns):
        for pat in patterns:
            hits = sorted(Path('.').glob(pat))
            if hits:
                return str(hits[0])
        return None

    # Carga automática desde la carpeta de ejecución. Si existen las planillas 01-05
    # junto a la app, también se fusionan automáticamente; si no, se pueden subir
    # desde el sidebar.
    main_auto = _first_existing(['ckd_data_v3_3.xlsx', 'CKD_DATA_v3_3.xlsx', 'CKD_DATA_v3.3.xlsx'])
    p01_auto = _first_existing(['01_otra_condicion_salud.xlsx', '01*condici*n*salud*.xlsx', '01*Condici*n*Salud*.xlsx', '01*.xlsx'])
    p02_auto = _first_existing(['02_antecedentes_familiares.xlsx', '02*Antec*.xlsx', '02*Fliares*.xlsx', '02*.xlsx'])
    p03_auto = _first_existing(['03_consumo_aines.xlsx', '03*AINE*.xlsx', '03*aine*.xlsx', '03*.xlsx'])
    p04_auto = _first_existing(['04_otros_diagnosticos.xlsx', '04*diagnost*.xlsx', '04*Diagnost*.xlsx', '04*.xlsx'])
    p05_auto = _first_existing(['05_evaluacion_medica.xlsx', '05*evaluacion*.xlsx', '05*Evaluacion*.xlsx', '05*.xlsx'])

    if main_auto is None:
        st.error(
            "No se encontró la base principal. Sube el archivo desde el sidebar.\n\n"
            "Nombres buscados: ckd_data_v3_3.xlsx, CKD_DATA_v3_3.xlsx o CKD_DATA_v3.3.xlsx."
        )
        st.stop()
    try:
        df_raw = load_data(main_auto, p01_auto, p02_auto, p03_auto, p04_auto, p05_auto)
        with st.sidebar.expander("Carga automática detectada", expanded=False):
            st.write(f"Base principal: {main_auto}")
            st.write(f"Planilla 01: {p01_auto or 'no detectada'}")
            st.write(f"Planilla 02: {p02_auto or 'no detectada'}")
            st.write(f"Planilla 03: {p03_auto or 'no detectada'}")
            st.write(f"Planilla 04: {p04_auto or 'no detectada'}")
            st.write(f"Planilla 05: {p05_auto or 'no detectada'}")
    except Exception as e:
        st.error(f"No se pudo cargar la base/planillas detectadas. Sube los archivos desde el sidebar.\n\n{e}")
        st.stop()

# ── Filtros ──────────────────────────────────────────────
st.sidebar.markdown("### Filtros")

age_min, age_max = st.sidebar.slider(
    "Rango de edad (años)",
    int(df_raw['Age'].min(skipna=True)), int(df_raw['Age'].max(skipna=True)),
    (int(df_raw['Age'].min(skipna=True)), int(df_raw['Age'].max(skipna=True))))

comm_counts = df_raw['Community_std'].value_counts()
all_comms = sorted(comm_counts[comm_counts >= 5].index.tolist())
sel_comms = st.sidebar.multiselect("Comunas (n≥5)", all_comms, default=all_comms)

ppoo_f = st.sidebar.multiselect("Pueblo originario",
    ['PPOO','No PPOO'], default=['PPOO','No PPOO'])
sex_f  = st.sidebar.multiselect("Sexo",
    ['Mujer','Hombre'], default=['Mujer','Hombre'])
rural_f = st.sidebar.multiselect("Ruralidad",
    ['Rural (≥30%)','Urbano (<30%)'], default=['Rural (≥30%)','Urbano (<30%)'])

with st.sidebar.expander("Filtros adicionales"):
    bmi_opts = [o for o in df_raw['BMI_cat'].dropna().unique()
                if o not in ('nan','None','')]
    bmi_f = st.multiselect("Estado nutricional", bmi_opts, default=bmi_opts)
    hta_r = st.radio("HTA autoreportada",
        ['Todos','Solo HTA','Solo sin HTA'], index=0)
    dm_r  = st.radio("DM autoreportada",
        ['Todos','Solo DM','Solo sin DM'], index=0)
    aines_r = st.radio("AINEs",
        ['Todos','Solo uso diario','Sin uso diario'], index=0)

# Aplicar filtros
df = df_raw[
    df_raw['Age'].between(age_min, age_max) &
    df_raw['Community_std'].isin(sel_comms) &
    (df_raw['PPOO_lbl'].isin(ppoo_f) | df_raw['PPOO_lbl'].isna()) &
    df_raw['Sex_lbl'].isin(sex_f) &
    (df_raw['Rural_lbl'].isin(rural_f) | df_raw['Rural_lbl'].isna()) &
    (df_raw['BMI_cat'].isin(bmi_f) | df_raw['BMI_cat'].isna())
].copy()
if hta_r == 'Solo HTA':   df = df[df['HTA_sr'] == 1]
elif hta_r == 'Solo sin HTA': df = df[df['HTA_sr'] == 0]
if dm_r  == 'Solo DM':    df = df[df['DM_sr'] == 1]
elif dm_r == 'Solo sin DM':   df = df[df['DM_sr'] == 0]
if aines_r == 'Solo uso diario': df = df[df['AINEs_diario_modelo'] == 1]
elif aines_r == 'Sin uso diario': df = df[df['AINEs_diario_modelo'] == 0]

st.sidebar.markdown("---")
st.sidebar.metric("N tras filtros", f"{len(df):,}",
                  f"{len(df)/len(df_raw)*100:.1f}% del total")
if len(df) < 20:
    st.warning(f"⚠️ Muestra muy pequeña ({len(df)} registros).")


# ═══════════════════════════════════════════════════════════
# MRP / SAE BAYESIANO — PREVALENCIA REGIONAL MODELADA
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Cargando resultados MRP/SAE…")
def load_mrp_results():
    base = Path(__file__).parent / "data_mrp"
    regional_file = base / "02_resumen_regional_mrp_v2c_calibrado_ENS.csv"
    comunal_final_file = base / "02_tabla_comunal_final_mrp_v2c.csv"
    comunal_resumen_file = base / "03_resumen_comunal_mrp_v2c_calibrado_ENS.csv"

    if not regional_file.exists():
        return None, None, base
    if comunal_final_file.exists():
        comunal_file = comunal_final_file
    elif comunal_resumen_file.exists():
        comunal_file = comunal_resumen_file
    else:
        return None, None, base

    return pd.read_csv(regional_file), pd.read_csv(comunal_file), base


def _fmt_pct_app(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"


def _fmt_num_app(x):
    try:
        return f"{float(x):,.0f}".replace(",", ".")
    except Exception:
        return "—"


def render_mrp_tab():
    st.header("Prevalencia regional y comunal modelada de ERC")

    st.info(
        "Estas estimaciones provienen de un modelo MRP/SAE Bayesiano calibrado "
        "externamente contra ENS. Las prevalencias comunales son estimaciones "
        "modeladas y no corresponden a prevalencias observadas directamente por comuna."
    )

    regional, comunal, base = load_mrp_results()
    if regional is None or comunal is None:
        st.warning("No se encontraron los archivos MRP en `data_mrp/`.")
        st.code(str(base))
        return

    indicador = st.selectbox(
        "Seleccione indicador",
        ["ERC G3a-G5", "ERC G3b-G5"],
        help="G3a-G5: eGFR <60. G3b-G5: eGFR <45.",
        key="mrp_indicador"
    )

    reg = regional[regional["indicador"].eq(indicador)].copy()
    com = comunal[comunal["indicador"].eq(indicador)].copy()

    if reg.empty or com.empty:
        st.warning("No hay datos para el indicador seleccionado.")
        return

    r = reg.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Población ≥20 años", _fmt_num_app(r.get("poblacion")))
    c2.metric("Prevalencia regional", _fmt_pct_app(r.get("prev_mediana")))
    c3.metric("IC95% prevalencia", f"{_fmt_pct_app(r.get('prev_p2_5'))} – {_fmt_pct_app(r.get('prev_p97_5'))}")
    c4.metric("Casos esperados", _fmt_num_app(r.get("casos_mediana")))

    st.caption(
        "Modelo principal: MRP v2c calibrado ENS. La campaña aporta gradientes "
        "de riesgo; ENS calibra el nivel regional; Censo y perfiles ENS/CASEN "
        "permiten postestratificación comunal."
    )

    st.subheader("Mapa comunal")
    if indicador == "ERC G3a-G5":
        img_file = base / "01b_mapa_prevalencia_erc_g3a_g5_mrp_v2c_con_etiquetas.png"
    else:
        img_file = base / "02b_mapa_prevalencia_erc_g3b_g5_mrp_v2c_con_etiquetas.png"

    if img_file.exists():
        st.image(str(img_file), use_container_width=True)
    else:
        st.warning("No se encontró el mapa etiquetado para este indicador.")
        st.code(str(img_file))

    st.subheader("Tabla comunal")

    rename_candidates = {
        "prevalencia_mediana": "prev_mediana",
        "prevalencia_ic95_inf": "prev_p2_5",
        "prevalencia_ic95_sup": "prev_p97_5",
        "casos_ic95_inf": "casos_p2_5",
        "casos_ic95_sup": "casos_p97_5",
    }
    for old, new in rename_candidates.items():
        if old in com.columns and new not in com.columns:
            com[new] = com[old]

    required = ["comuna", "poblacion", "prev_mediana", "prev_p2_5", "prev_p97_5",
                "casos_mediana", "casos_p2_5", "casos_p97_5"]
    missing_cols = [c for c in required if c not in com.columns]
    if missing_cols:
        st.warning(f"Faltan columnas esperadas en tabla comunal: {missing_cols}")
        st.dataframe(com, use_container_width=True, hide_index=True)
        return

    if "exceso_relativo_vs_region" not in com.columns:
        com["exceso_relativo_vs_region"] = com["prev_mediana"] / float(r.get("prev_mediana"))

    com_view = com.copy()
    com_view["Prevalencia"] = com_view["prev_mediana"].apply(_fmt_pct_app)
    com_view["IC95% prev."] = com_view.apply(lambda x: f"{_fmt_pct_app(x['prev_p2_5'])} – {_fmt_pct_app(x['prev_p97_5'])}", axis=1)
    com_view["Casos esperados"] = com_view["casos_mediana"].apply(_fmt_num_app)
    com_view["IC95% casos"] = com_view.apply(lambda x: f"{_fmt_num_app(x['casos_p2_5'])} – {_fmt_num_app(x['casos_p97_5'])}", axis=1)
    com_view["Razón vs región"] = com_view["exceso_relativo_vs_region"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    sort_opt = st.radio("Ordenar tabla por", ["Mayor prevalencia", "Mayor número de casos", "Nombre comuna"], horizontal=True, key="mrp_sort")
    if sort_opt == "Mayor prevalencia":
        com_view = com_view.sort_values("prev_mediana", ascending=False)
    elif sort_opt == "Mayor número de casos":
        com_view = com_view.sort_values("casos_mediana", ascending=False)
    else:
        com_view = com_view.sort_values("comuna")

    st.dataframe(
        com_view[["comuna", "poblacion", "Prevalencia", "IC95% prev.", "Casos esperados", "IC95% casos", "Razón vs región"]]
        .rename(columns={"comuna": "Comuna", "poblacion": "Población ≥20"}),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Ranking comunal")
    top_n = st.slider("Número de comunas a mostrar", min_value=5, max_value=32, value=10, key="mrp_topn")

    top_prev = com.sort_values("prev_mediana", ascending=False).head(top_n).copy()
    fig_prev = px.bar(
        top_prev.sort_values("prev_mediana"),
        x="prev_mediana",
        y="comuna",
        orientation="h",
        text=top_prev.sort_values("prev_mediana")["prev_mediana"].apply(lambda x: f"{x*100:.1f}%"),
        labels={"prev_mediana": "Prevalencia modelada", "comuna": "Comuna"},
        title=f"Top {top_n} comunas por prevalencia modelada — {indicador}"
    )
    fig_prev.update_layout(height=max(360, top_n * 28), margin=dict(l=10, r=10, t=45, b=20))
    fig_prev.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_prev, use_container_width=True)

    top_cases = com.sort_values("casos_mediana", ascending=False).head(top_n).copy()
    fig_cases = px.bar(
        top_cases.sort_values("casos_mediana"),
        x="casos_mediana",
        y="comuna",
        orientation="h",
        text=top_cases.sort_values("casos_mediana")["casos_mediana"].apply(lambda x: f"{x:,.0f}".replace(",", ".")),
        labels={"casos_mediana": "Casos esperados", "comuna": "Comuna"},
        title=f"Top {top_n} comunas por casos esperados — {indicador}"
    )
    fig_cases.update_layout(height=max(360, top_n * 28), margin=dict(l=10, r=10, t=45, b=20))
    st.plotly_chart(fig_cases, use_container_width=True)

    with st.expander("Nota metodológica", expanded=False):
        st.markdown(
            "**Interpretación:**\n\n"
            "Las estimaciones comunales son predicciones modeladas mediante MRP/SAE "
            "y calibradas contra la prevalencia regional ENS. En comunas sin datos "
            "directos de campaña, la estimación depende de la estructura poblacional "
            "censal, perfiles de riesgo derivados de ENS, covariables comunales y "
            "pooling jerárquico.\n\n"
            "**Uso recomendado:** identificación territorial exploratoria, priorización "
            "programática y generación de hipótesis. No debe interpretarse como "
            "estimación comunal observada ni representativa directa."
        )


# ═══════════════════════════════════════════════════════════
# ENCABEZADO
# ═══════════════════════════════════════════════════════════
st.title("Estudio de prevalencia de ERC — Región de la Araucanía")
st.caption("Dashboard interactivo v4.2.14 · eGFR CKD-EPI 2021 · Muestra de screening comunitario")

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
(tab_overview, tab_prev, tab_strat, tab_subgrp,
 tab_risk, tab_score, tab_concord, tab_lit, tab_data, tab_mrp) = st.tabs([
    "🏠 Resumen", "📊 Pesquisa ERC", "🔬 Estratificación",
    "🧩 Carga de riesgo", "⚠️ Factores riesgo", "📈 Score",
    "🔄 Concordancia", "📖 Literatura", "📋 Datos", "🗺️ Prevalencia modelada",
])

# ═══════════════════════════════════════════════════════════
# TAB MRP — PREVALENCIA REGIONAL MODELADA
# ═══════════════════════════════════════════════════════════
with tab_mrp:
    render_mrp_tab()



# ═══════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ═══════════════════════════════════════════════════════════
with tab_overview:
    st.header("Características de la muestra")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("N analítico", f"{len(df):,}")
    c2.metric("Edad media", f"{df['Age'].mean():.1f} a")
    c3.metric("% Mujeres",
              f"{(df['Sex_lbl']=='Mujer').sum()/len(df)*100:.1f}%")
    c4.metric("% PPOO",
              f"{(df['PPOO_n']==1).sum()/df['PPOO_n'].notna().sum()*100:.1f}%"
              if df['PPOO_n'].notna().any() else "—")
    ckd_n = int((df['CKD60']==1).sum()); ckd_N = int(df['CKD60'].notna().sum())
    p,l,h = wilson(ckd_n,ckd_N)
    c5.metric("Prev. ERC (CKD-EPI 2021)",
              f"{p:.1f}%" if pd.notna(p) else "—",
              f"IC {l:.1f}–{h:.1f}")

    st.markdown("### Prevalencias clave")
    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,var,hlp) in zip([c1,c2,c3,c4],[
        ("ERC eGFR<60","CKD60","Ecuación CKD-EPI 2021"),
        ("ERC ampliada","CKD_exp","eGFR<60 O proteinuria+"),
        ("HTA medida","HTA_meas","PA ≥130/80 mmHg"),
        ("Proteinuria +","prot_pos","Tira reactiva positiva"),
    ]):
        n=int((df[var]==1).sum()); N=int(df[var].notna().sum())
        pv,lo,hi=wilson(n,N)
        col.metric(lbl, f"{pv:.1f}%" if pd.notna(pv) else "—",
                   f"IC {lo:.1f}–{hi:.1f}" if pd.notna(lo) else "", help=hlp)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### Pirámide etaria")
        pir = df.dropna(subset=['Age_grp','Sex_lbl'])
        order = ['<30','30-44','45-59','60-74','≥75']
        ct = pd.crosstab(pir['Age_grp'], pir['Sex_lbl']).reindex(order).fillna(0)
        fig = go.Figure()
        if 'Hombre' in ct.columns:
            fig.add_trace(go.Bar(y=ct.index, x=-ct['Hombre'], name='Hombres',
                orientation='h', marker_color=C_MAIN,
                customdata=ct['Hombre'],
                hovertemplate='Hombres: %{customdata}<extra></extra>'))
        if 'Mujer' in ct.columns:
            fig.add_trace(go.Bar(y=ct.index, x=ct['Mujer'], name='Mujeres',
                orientation='h', marker_color=C_ACC,
                hovertemplate='Mujeres: %{x}<extra></extra>'))
        mx = max(ct.max().max(),1)
        fig.update_layout(
            barmode='relative',
            xaxis=dict(tickvals=[-mx,-mx/2,0,mx/2,mx],
                       ticktext=[str(int(mx)),str(int(mx/2)),'0',str(int(mx/2)),str(int(mx))]),
            height=360, margin=dict(l=10,r=10,t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Distribución eGFR (CKD-EPI 2021)")
        if df['eGFR'].notna().any():
            fig = px.histogram(df.dropna(subset=['eGFR']), x='eGFR', nbins=50,
                               color_discrete_sequence=[C_TEAL])
            fig.add_vline(x=60, line_dash='dash', line_color=C_ACC,
                          annotation_text='60 mL/min')
            fig.add_vline(x=90, line_dash='dot', line_color=C_GRAY,
                          annotation_text='90 mL/min')
            fig.update_layout(xaxis_title='eGFR CKD-EPI 2021 (mL/min/1.73m²)',
                yaxis_title='N', height=360,
                margin=dict(l=10,r=10,t=10,b=20))
            st.plotly_chart(fig, use_container_width=True)
            if df['eGFR_orig'].notna().any():
                st.caption(
                    f"⚡ CKD-EPI 2021 vs eGFR original: "
                    f"casos eGFR<60 = {int((df['eGFR']<60).sum())} vs "
                    f"{int((df['eGFR_orig']<60).sum())} "
                    f"(+{int((df['eGFR']<60).sum()-( df['eGFR_orig']<60).sum())} "
                    f"con nueva ecuación)")

    # BMI detallado
    st.markdown("### Estado nutricional — detalle por clase de obesidad")
    if df['BMI_cat'].notna().any():
        order_bmi = ['Bajo peso','Peso normal','Sobrepeso',
                     'Obesidad clase I','Obesidad clase II','Obesidad clase III']
        bmi_counts = df['BMI_cat'].value_counts().reindex(order_bmi).fillna(0)
        bmi_pcts = bmi_counts / bmi_counts.sum() * 100
        bmi_prev = []
        for cat in order_bmi:
            sv = df[df['BMI_cat']==cat].dropna(subset=['CKD60'])
            n=int((sv['CKD60']==1).sum()); N=int(len(sv))
            bmi_prev.append(wilson(n,N)[0] if N>=5 else np.nan)
        bmi_colors=['#88C640','#4DAF4A','#FFDD00','#F28F00','#E83800','#9E0000']
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=order_bmi, y=bmi_counts.values,
            marker_color=bmi_colors, opacity=0.8,
            text=[f"{int(n)}" for n in bmi_counts.values],
            textposition='outside', name='N'), secondary_y=False)
        fig.add_trace(go.Scatter(x=order_bmi, y=bmi_prev,
            mode='lines+markers', line=dict(color=C_ACC,width=2.5),
            marker=dict(size=10), name='Prev ERC %',
            text=[f"{p:.1f}%" if pd.notna(p) else "" for p in bmi_prev],
            textposition='top center'), secondary_y=True)
        fig.update_yaxes(title_text="N pacientes", secondary_y=False)
        fig.update_yaxes(title_text="Positividad ERC (%)", secondary_y=True)
        fig.update_layout(height=370, margin=dict(l=10,r=10,t=10,b=20),
                          legend=dict(orientation='h',y=1.08))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — PESQUISA ERC
# ═══════════════════════════════════════════════════════════
with tab_prev:
    st.header("Positividad observada de ERC — definiciones operacionales")

    rows=[]
    for lbl,var in [
        ("ERC presuntiva — eGFR CKD-EPI 2021 <60","CKD60"),
        ("ERC ampliada (eGFR<60 O proteinuria+)","CKD_exp"),
        ("Proteinuria positiva aislada","prot_pos"),
        ("Hematuria positiva aislada","blood_pos"),
    ]:
        n=int((df[var]==1).sum()); N=int(df[var].notna().sum())
        p,l,h=wilson(n,N)
        rows.append([lbl,n,N,f"{p:.1f}%" if pd.notna(p) else "—",
                     f"{l:.1f}–{h:.1f}" if pd.notna(l) else "—"])
    st.dataframe(pd.DataFrame(rows,
        columns=['Definición','n','N','Positividad','IC95%']),
        use_container_width=True, hide_index=True)

    # Comparación ecuaciones
    if df['eGFR_orig'].notna().any():
        st.info(
            f"**Impacto ecuación CKD-EPI 2021:** eGFR<60 original = "
            f"{int((df['eGFR_orig']<60).sum())} ({(df['eGFR_orig']<60).sum()/df['eGFR_orig'].notna().sum()*100:.1f}%) → "
            f"CKD-EPI 2021 = {int((df['eGFR']<60).sum())} ({(df['eGFR']<60).sum()/df['eGFR'].notna().sum()*100:.1f}%) "
            f"(sin cap en 90, 14 casos reclasificados a <60 en la muestra completa)"
        )

    st.markdown("### Distribución por estadio KDIGO G")
    col1,col2 = st.columns([2,1])
    with col1:
        order=['G1','G2','G3a','G3b','G4','G5']
        counts=df['KDIGO_G'].value_counts().reindex(order).fillna(0)
        total=counts.sum()
        fig=go.Figure()
        fig.add_trace(go.Bar(
            x=order, y=counts.values,
            marker_color=[KDIGO_COLORS[g] for g in order],
            text=[f"{int(c)}<br>({c/total*100:.1f}%)" for c in counts.values],
            textposition='outside'))
        fig.update_layout(xaxis_title='Estadio KDIGO G',
            yaxis_title='N pacientes', showlegend=False,
            height=400, margin=dict(l=10,r=10,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Riesgo combinado KDIGO**")
        rc=df['KDIGO_risk'].value_counts().reindex(
            ['Bajo','Moderado','Alto','Muy alto']).fillna(0)
        fig=go.Figure(go.Pie(
            labels=rc.index, values=rc.values,
            marker=dict(colors=[RISK_COLORS[k] for k in rc.index]),
            hole=0.45, textinfo='label+percent', textposition='outside'))
        fig.update_layout(height=340,margin=dict(l=10,r=10,t=10,b=10),showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        hi=int(df['KDIGO_risk'].isin(['Alto','Muy alto']).sum())
        tot=int(df['KDIGO_risk'].notna().sum())
        st.metric("Alto + Muy alto", f"{hi}",
                  f"{hi/tot*100:.1f}% del total" if tot else "")

    st.markdown("### Matriz KDIGO (eGFR × Albuminuria)")
    sub=df.dropna(subset=['KDIGO_G','Alb_cat'])
    if len(sub) > 0:
        tab_m = pd.crosstab(sub['KDIGO_G'],sub['Alb_cat']).reindex(
            index=['G1','G2','G3a','G3b','G4','G5'],
            columns=['A1','A2','A3']).fillna(0).astype(int)
        risk_grid=pd.DataFrame([
            ['Bajo','Moderado','Alto'],['Bajo','Moderado','Alto'],
            ['Moderado','Alto','Muy alto'],['Alto','Muy alto','Muy alto'],
            ['Muy alto','Muy alto','Muy alto'],['Muy alto','Muy alto','Muy alto'],
        ], index=tab_m.index, columns=tab_m.columns)

        # Color = categoría de riesgo KDIGO; intensidad = positividad de la celda.
        total_m = tab_m.values.sum()
        pct_m = tab_m / total_m * 100 if total_m else tab_m.astype(float)
        max_pct = pct_m.values.max() if total_m else 0
        x_labels = ['A1 (Neg)', 'A2 (30 mg/dL)', 'A3 (≥100)']
        y_labels = list(tab_m.index)
        fig = go.Figure()
        for iy, g in enumerate(y_labels):
            for ix, a in enumerate(tab_m.columns):
                n_cell = int(tab_m.loc[g, a])
                pct_cell = float(pct_m.loc[g, a]) if total_m else 0.0
                risk = risk_grid.loc[g, a]
                if n_cell == 0:
                    fill = 'white'
                    txt = ''
                    hover = f'{g} + {x_labels[ix]}: sin datos'
                else:
                    intensity = 0.25 + 0.75*(pct_cell/max_pct) if max_pct > 0 else 0.25
                    fill = blend_with_white(RISK_COLORS[risk], intensity)
                    txt = f'<b>{n_cell}</b><br>{pct_cell:.1f}%'
                    hover = f'{g} + {x_labels[ix]}: {n_cell} pacientes ({pct_cell:.1f}%)<br>Riesgo KDIGO: {risk}'
                fig.add_shape(type='rect', x0=ix-0.5, x1=ix+0.5, y0=iy-0.5, y1=iy+0.5,
                              line=dict(color='#FFFFFF', width=2), fillcolor=fill)
                fig.add_annotation(x=ix, y=iy, text=txt, showarrow=False,
                                   font=dict(size=13, color='#222222'))
                fig.add_trace(go.Scatter(x=[ix], y=[iy], mode='markers',
                                         marker=dict(size=40, color='rgba(0,0,0,0)'),
                                         hovertemplate=hover + '<extra></extra>', showlegend=False))
        fig.update_xaxes(tickmode='array', tickvals=list(range(len(x_labels))), ticktext=x_labels,
                         title='Albuminuria', showgrid=False, zeroline=False)
        fig.update_yaxes(tickmode='array', tickvals=list(range(len(y_labels))), ticktext=y_labels,
                         title='Estadio eGFR', autorange='reversed', showgrid=False, zeroline=False)
        fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=20), plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Color base según riesgo KDIGO; intensidad proporcional a la positividad dentro de la matriz. Celdas sin datos quedan en blanco.')



# ═══════════════════════════════════════════════════════════
# TAB 3 — ESTRATIFICACIÓN
# ═══════════════════════════════════════════════════════════
with tab_strat:
    st.header("Positividad observada de ERC estratificada")

    c_ctrl, c_plot = st.columns([1,3])
    with c_ctrl:
        var_opts = {
            'Age_grp':'Grupo etario','Sex_lbl':'Sexo','PPOO_lbl':'Pueblo originario',
            'Community_std':'Comuna','BMI_cat':'Estado nutricional (detallado)',
            'BP_Final':'Categoría PA','HTA_sr':'HTA autoreportada',
            'DM_sr':'DM autoreportada','FamHx_ERC':'Antec familiar ERC',
            'Tabaco':'Tabaquismo','OH_b':'Consumo alcohol',
            'Rural_lbl':'Ruralidad',
        }
        # Suma variables binarias estandarizadas 01-05 con frecuencia suficiente.
        var_opts.update({c: humanize_varname(c) for c in standardized_binary_columns(df)})
        var_s = st.selectbox("Variable principal eje X", list(var_opts.keys()),
                             format_func=lambda x: var_opts.get(x,x))
        var2_options = ['(Ninguna)'] + [v for v in var_opts.keys() if v != var_s]
        var2_s = st.selectbox("Segunda variable opcional", var2_options,
                              format_func=lambda x: 'Sin segunda variable' if x=='(Ninguna)' else var_opts.get(x,x))
        outcome_s = st.radio("Definición ERC",
            ['CKD60','CKD_exp'],
            format_func=lambda x: 'Estricta (CKD-EPI 2021 <60)' if x=='CKD60'
                                  else 'Ampliada (incl. proteinuria)')
        min_n = st.number_input("N mínimo", 1, 100, 5)
        st.caption("La segunda variable muestra barras agrupadas en paralelo para comparar positividad por subgrupo.")

    with c_plot:
        group_cols = [var_s] if var2_s == '(Ninguna)' else [var_s, var2_s]
        sub = df.dropna(subset=group_cols + [outcome_s]).copy()
        if len(sub) < 5:
            st.warning("Sin datos suficientes con estos filtros.")
        else:
            ordered={'Age_grp':['<30','30-44','45-59','60-74','≥75'],
                     'BMI_cat':['Bajo peso','Peso normal','Sobrepeso',
                                'Obesidad clase I','Obesidad clase II','Obesidad clase III']}
            if var2_s == '(Ninguna)':
                grp = sub.groupby(var_s).agg(n_ckd=(outcome_s,'sum'), N=(outcome_s,'size'))
                grp = grp[grp['N'] >= min_n]
                grp['prev'] = grp['n_ckd']/grp['N']*100
                grp[['lo','hi']] = grp.apply(lambda r: pd.Series(wilson(r['n_ckd'],r['N'])[1:]), axis=1)
                if var_s in ordered:
                    grp=grp.reindex(ordered[var_s]).dropna(how='all')
                else:
                    grp=grp.sort_values('prev',ascending=False)
                grp=grp.reset_index()

                fig=go.Figure()
                fig.add_trace(go.Bar(
                    x=grp[var_s].astype(str), y=grp['prev'],
                    error_y=dict(type='data',array=grp['hi']-grp['prev'],
                                 arrayminus=grp['prev']-grp['lo']),
                    marker_color=C_MAIN))
                labels_s=[f"{p:.1f}%<br>(n={int(N)})" for p,N in zip(grp['prev'],grp['N'])]
                anns_s=bar_annotations(grp[var_s].astype(str).tolist(), grp['prev'].tolist(), grp['hi'].tolist(), labels_s)
                fig.update_layout(yaxis_title='Positividad ERC (%)',
                    height=420, margin=dict(l=10,r=10,t=50,b=20),
                    showlegend=False, annotations=anns_s,
                    yaxis=dict(range=[0, max(grp['hi'].max()*1.35, 5)]))
                st.plotly_chart(fig, use_container_width=True)

                test = smart_categorical_test(sub, var_s, outcome_s)
                if pd.notna(test['p']):
                    note = f" · {test['note']}" if test.get('note') else ""
                    st.caption(f"**Test global:** {test['method']}, p = {format_p(test['p'])} "
                               f"{'✓ diferencias entre grupos' if test['p']<0.05 else '✗ sin evidencia de diferencias'}{note}. "
                               "Como el desenlace es binario, se comparan proporciones; la positividad graficada es el resumen por grupo.")
                    ph = pairwise_categorical_posthoc(sub, var_s, outcome_s, min_n=min_n)
                    if test['p'] < 0.05 and len(ph):
                        sig_ph = ph[ph['Significativo']].copy()
                        if len(sig_ph):
                            st.caption("**Post hoc Holm:** diferencias significativas en " +
                                       "; ".join(sig_ph['Comparación'].astype(str).head(6)) +
                                       ("; …" if len(sig_ph)>6 else ""))
                tbl=grp[[var_s,'n_ckd','N','prev','lo','hi']].copy()
                tbl.columns=['Estrato','n ERC','N','Prev %','IC inf','IC sup']
                tbl[['Prev %','IC inf','IC sup']]=tbl[['Prev %','IC inf','IC sup']].round(1)
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                if pd.notna(test.get('p', np.nan)) and test['p'] < 0.05:
                    ph_show = pairwise_categorical_posthoc(sub, var_s, outcome_s, min_n=min_n)
                    if len(ph_show):
                        ph_show['Δ positividad pp'] = ph_show['Δ positividad pp'].round(1)
                        ph_show['p'] = ph_show['p'].round(4)
                        ph_show['p ajustado Holm'] = ph_show['p ajustado Holm'].round(4)
                        st.markdown("**Comparaciones post hoc entre grupos**")
                        st.dataframe(style_sig(ph_show), use_container_width=True, hide_index=True)
            else:
                grp = sub.groupby([var_s, var2_s]).agg(n_ckd=(outcome_s,'sum'), N=(outcome_s,'size')).reset_index()
                grp = grp[grp['N'] >= min_n].copy()
                if len(grp) == 0:
                    st.warning("No hay combinaciones con el N mínimo seleccionado.")
                else:
                    grp['prev'] = grp['n_ckd']/grp['N']*100
                    grp[['lo','hi']] = grp.apply(lambda r: pd.Series(wilson(r['n_ckd'],r['N'])[1:]), axis=1)
                    if var_s in ordered:
                        grp[var_s] = pd.Categorical(grp[var_s], categories=ordered[var_s], ordered=True)
                        grp = grp.sort_values([var_s, var2_s])
                    else:
                        order_x = grp.groupby(var_s)['prev'].mean().sort_values(ascending=False).index.tolist()
                        grp[var_s] = pd.Categorical(grp[var_s], categories=order_x, ordered=True)
                        grp = grp.sort_values([var_s, var2_s])
                    fig = go.Figure()

                    # Gráfico agrupado construido manualmente para controlar la posición
                    # de las etiquetas. En Plotly, textposition='outside' ubica el texto
                    # en el extremo de la barra y puede superponerse con las barras de error.
                    # Aquí las etiquetas se agregan como anotaciones por encima del IC95%.
                    x_order = grp[var_s].astype(str).drop_duplicates().tolist()
                    levels2 = grp[var2_s].astype(str).drop_duplicates().tolist()
                    colors2 = px.colors.qualitative.Plotly

                    label_y_values = []
                    for j, level in enumerate(levels2):
                        sg = grp[grp[var2_s].astype(str) == level].copy()
                        color = colors2[j % len(colors2)]
                        fig.add_trace(go.Bar(
                            x=sg[var_s].astype(str),
                            y=sg['prev'],
                            name=str(level),
                            offsetgroup=str(level),
                            marker_color=color,
                            error_y=dict(
                                type='data',
                                array=sg['hi'] - sg['prev'],
                                arrayminus=sg['prev'] - sg['lo'],
                                visible=True
                            ),
                            customdata=np.stack([sg['N'], sg['n_ckd'], sg['hi']], axis=-1),
                            hovertemplate=(
                                f"{var_opts.get(var_s, var_s)}: %{{x}}<br>"
                                f"{var_opts.get(var2_s, var2_s)}: {level}<br>"
                                "Positividad: %{y:.1f}%<br>"
                                "N: %{customdata[0]}<br>"
                                "n ERC: %{customdata[1]}<br>"
                                "IC95% sup.: %{customdata[2]:.1f}%"
                                "<extra></extra>"
                            )
                        ))

                        # Desplazamiento horizontal en píxeles para que las etiquetas
                        # queden centradas sobre cada barra del grupo.
                        xshift = (j - (len(levels2) - 1) / 2) * 42
                        for _, r in sg.iterrows():
                            y_label = float(r['hi']) + max(float(grp['hi'].max()) * 0.055, 1.2)
                            label_y_values.append(y_label)
                            fig.add_annotation(
                                x=str(r[var_s]),
                                y=y_label,
                                xref='x',
                                yref='y',
                                text=f"{r['prev']:.1f}%<br>n={int(r['N'])}",
                                showarrow=False,
                                xanchor='center',
                                yanchor='bottom',
                                xshift=xshift,
                                font=dict(size=11, color='#5F6680'),
                                bgcolor='rgba(255,255,255,0.75)',
                                borderpad=1
                            )

                    ymax = max(
                        max(label_y_values) * 1.18 if label_y_values else 0,
                        grp['hi'].max() * 1.65,
                        8
                    )

                    fig.update_layout(
                        barmode='group',
                        yaxis_title='Positividad ERC (%)',
                        xaxis_title=var_opts.get(var_s, var_s),
                        height=520,
                        margin=dict(l=10, r=10, t=95, b=70),
                        legend_title=var_opts.get(var2_s, var2_s),
                        yaxis=dict(range=[0, ymax]),
                    )
                    fig.update_xaxes(categoryorder='array', categoryarray=x_order)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Barras agrupadas por la segunda variable. La prueba global formal se mantiene para la variable principal; la comparación multivariable se recomienda interpretarla descriptivamente.")
                    test = smart_categorical_test(sub, var_s, outcome_s)
                    if pd.notna(test['p']):
                        st.caption(f"**Test global variable principal:** {test['method']}, p = {format_p(test['p'])}.")
                    tbl=grp[[var_s,var2_s,'n_ckd','N','prev','lo','hi']].copy()
                    tbl.columns=['Estrato principal','Subgrupo','n ERC','N','Prev %','IC inf','IC sup']
                    tbl[['Prev %','IC inf','IC sup']]=tbl[['Prev %','IC inf','IC sup']].round(1)
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # PPOO × Edad (estricta y ampliada)
    st.markdown("---")
    st.markdown("### PPOO × Edad — definición estricta vs ampliada")
    col1,col2 = st.columns(2)
    for col,out_v,lbl_v in [(col1,'CKD60','Estricta'),
                             (col2,'CKD_exp','Ampliada')]:
        with col:
            sub2=df.dropna(subset=['Age_grp','PPOO_lbl',out_v])
            rows2=[]
            for g in ['<30','30-44','45-59','60-74','≥75']:
                for ppoo in ['PPOO','No PPOO']:
                    sv=sub2[(sub2['Age_grp']==g)&(sub2['PPOO_lbl']==ppoo)]
                    if len(sv)<5: continue
                    n=int((sv[out_v]==1).sum())
                    p,lo,hi=wilson(n,len(sv))
                    rows2.append({'Edad':g,'PPOO':ppoo,'prev':p,'lo':lo,'hi':hi,'N':len(sv)})
            if rows2:
                pdf=pd.DataFrame(rows2)
                fig=px.bar(pdf,x='Edad',y='prev',color='PPOO',barmode='group',
                    color_discrete_map={'PPOO':C_PPOO,'No PPOO':C_NOPP},
                    error_y=pdf['hi']-pdf['prev'],
                    category_orders={'Edad':['<30','30-44','45-59','60-74','≥75']},
                    hover_data=['N'], title=f"Definición {lbl_v}")
                fig.update_layout(yaxis_title='Positividad (%)',
                    height=380, margin=dict(l=10,r=10,t=30,b=20))
                col.plotly_chart(fig, use_container_width=True)

    # Ruralidad + comunas
    st.markdown("### Positividad comunal y ruralidad")
    com_data=[]
    for com in all_comms:
        sv=df[df['Community_std']==com].dropna(subset=['CKD60'])
        if len(sv)<10: continue
        n=int((sv['CKD60']==1).sum()); N=int(len(sv))
        p,lo,hi=wilson(n,N)
        r=RURALIDAD_CANONICA.get(com,np.nan)
        com_data.append({'Comuna':com,'N':N,'Prev':p,'lo':lo,'hi':hi,
                         'Ruralidad':r,'Rural':r>=30 if pd.notna(r) else False})
    if com_data:
        cdf=pd.DataFrame(com_data).sort_values('Ruralidad',ascending=False)
        fig=go.Figure()
        fig.add_trace(go.Bar(
            x=cdf['Comuna'], y=cdf['Prev'],
            error_y=dict(type='data',array=cdf['hi']-cdf['Prev'],
                         arrayminus=cdf['Prev']-cdf['lo']),
            marker_color=[C_RURAL if r else C_URBAN for r in cdf['Rural']],
            text=[f"{p:.1f}%<br>({r:.0f}%R)" for p,r in
                  zip(cdf['Prev'],cdf['Ruralidad'].fillna(0))],
            textposition='outside',
            hovertemplate='%{x}<br>ERC: %{y:.1f}%<extra></extra>'))
        fig.update_layout(
            xaxis_title='% Positividad ERC (% Ruralidad de la comuna)',
            yaxis_title='Positividad eGFR<60 (%)',
            height=420, margin=dict(l=10,r=10,t=10,b=60),
            annotations=[dict(text='🟫 Rural ≥30%  🔵 Urbano <30%',
                x=0.5,y=-0.20,xref='paper',yref='paper',showarrow=False,
                font=dict(size=11))])
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — SUBGRUPOS
# ═══════════════════════════════════════════════════════════
with tab_subgrp:
    st.header("Carga y combinaciones de factores de riesgo")

    # Carga acumulada
    st.markdown("### Carga acumulada de factores de riesgo")
    st.caption("Factores: edad≥60, HTA medida, DM (autoreporte o HGT>200), obesidad, antec. familiar ERC")

    # Cada factor se codifica como 0/1 y luego se suma.  La categoría final agrupa
    # explícitamente a todos los pacientes con 3, 4 o 5 factores presentes.
    rf_components = pd.DataFrame({
        'Edad ≥60': (df['Age'] >= 60).astype(float),
        'HTA medida': df['HTA_meas'].fillna(0).astype(float),
        'DM autorep/HGT>200': ((df['DM_sr'] == 1) | (df['Glc200'] == 1)).astype(float),
        'Obesidad': df['Obesity'].fillna(0).astype(float),
        'Antecedente familiar ERC': df['FamHx_ERC'].fillna(0).astype(float),
    }, index=df.index)

    df['n_rf'] = rf_components.sum(axis=1)
    rf_order = ['0', '1', '2', '≥3']

    # Crear categorías de carga acumulada de forma robusta.
    # Se evita np.select para no mezclar tipos texto/NaN y se fuerza que
    # la categoría ≥3 exista siempre como nivel del eje X, aunque alguna
    # combinación de filtros deje pocos o ningún caso en esa categoría.
    df['RF_lbl'] = pd.cut(
        df['n_rf'],
        bins=[-0.1, 0.5, 1.5, 2.5, np.inf],
        labels=rf_order,
        ordered=True
    )

    sub = df.dropna(subset=['CKD60', 'RF_lbl']).copy()
    sub['RF_lbl'] = pd.Categorical(sub['RF_lbl'].astype(str), categories=rf_order, ordered=True)
    if len(sub) > 0:
        rows3 = []
        for lbl in rf_order:
            sg = sub[sub['RF_lbl'] == lbl]
            n = int((sg['CKD60'] == 1).sum()) if len(sg) else 0
            if len(sg):
                p, lo, hi = wilson(n, len(sg))
            else:
                p, lo, hi = (0, 0, 0)
            rows3.append({'Score': lbl, 'prev': p, 'lo': lo, 'hi': hi, 'N': len(sg), 'n ERC': n})

        sdf = pd.DataFrame(rows3)
        sdf['Score'] = pd.Categorical(sdf['Score'].astype(str), categories=rf_order, ordered=True)
        sdf = sdf.sort_values('Score').reset_index(drop=True)
        if len(sdf):
            # Usar posiciones numéricas evita que Plotly interprete mal la etiqueta "≥3"
            # y asegura que la cuarta barra se dibuje siempre. El eje se rotula con ticktext.
            sdf['xpos'] = range(len(rf_order))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sdf['xpos'].astype(int).tolist(),
                y=sdf['prev'].astype(float).tolist(),
                width=0.62,
                error_y=dict(
                    type='data',
                    array=(sdf['hi'] - sdf['prev']).astype(float).tolist(),
                    arrayminus=(sdf['prev'] - sdf['lo']).astype(float).tolist()
                ),
                marker_color=[RISK_COLORS['Bajo'], RISK_COLORS['Moderado'],
                              RISK_COLORS['Alto'], RISK_COLORS['Muy alto']],
                customdata=sdf[['Score', 'N', 'n ERC', 'lo', 'hi']].astype(str).values,
                hovertemplate=(
                    'N° factores: %{customdata[0]}<br>'
                    'Positividad: %{y:.1f}%<br>'
                    'N: %{customdata[1]}<br>'
                    'n ERC: %{customdata[2]}<br>'
                    'IC95%: %{customdata[3]}–%{customdata[4]}<extra></extra>'
                )
            ))

            anns_rf = bar_annotations(
                sdf['xpos'].astype(int).tolist(),
                sdf['prev'].tolist(),
                sdf['hi'].tolist(),
                [f"{p:.1f}%<br>n={N}" for p, N in zip(sdf['prev'], sdf['N'])]
            )

            y_max_rf = max(float(sdf['hi'].max()) * 1.35, 5)
            fig.update_layout(
                xaxis_title='N° factores de riesgo presentes',
                yaxis_title='Positividad ERC (%)',
                annotations=anns_rf,
                yaxis=dict(range=[0, y_max_rf]),
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(rf_order))),
                    ticktext=rf_order,
                    range=[-0.5, len(rf_order) - 0.5]
                ),
                height=380,
                margin=dict(l=10, r=10, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            n_ge3 = int(sdf.loc[sdf['Score'].astype(str).eq('≥3'), 'N'].sum())
            st.caption(f"La columna **≥3** agrupa a quienes presentan tres o más factores de riesgo simultáneos. N ≥3 = {n_ge3}.")

            tbl_rf = sdf[['Score', 'n ERC', 'N', 'prev', 'lo', 'hi']].copy()
            tbl_rf.columns = ['N° factores', 'n ERC', 'N', 'Prev %', 'IC inf', 'IC sup']
            tbl_rf[['Prev %', 'IC inf', 'IC sup']] = tbl_rf[['Prev %', 'IC inf', 'IC sup']].round(1)
            with st.expander("Ver tabla de carga acumulada", expanded=False):
                st.dataframe(tbl_rf, use_container_width=True, hide_index=True)

            test_rf = smart_categorical_test(sub, 'RF_lbl', 'CKD60')
            if pd.notna(test_rf['p']):
                note = f" · {test_rf['note']}" if test_rf.get('note') else ""
                st.caption(f"**Test global:** {test_rf['method']}, p = {format_p(test_rf['p'])} "
                           f"{'✓ la Positividad cambia con la carga de factores' if test_rf['p']<0.05 else '✗ sin evidencia de diferencias'}{note}.")
                ph_rf = pairwise_categorical_posthoc(sub, 'RF_lbl', 'CKD60', min_n=5)
                if test_rf['p'] < 0.05 and len(ph_rf):
                    ph_rf['Δ Positividad pp'] = ph_rf['Δ Positividad pp'].round(1)
                    ph_rf['p'] = ph_rf['p'].round(4)
                    ph_rf['p ajustado Holm'] = ph_rf['p ajustado Holm'].round(4)
                    st.dataframe(style_sig(ph_rf), use_container_width=True, hide_index=True)

    # Combinaciones clínicas
    st.markdown("### Combinaciones clínicas críticas")
    st.caption(
        "Los subgrupos se definen solo por antecedentes, exposiciones o condiciones basales. "
        "La variable eGFR<60 no se utiliza para formar combinaciones, porque corresponde al desenlace ERC."
    )
    combos=[
        ("Sin factores conocidos",
         (df['Age']<60)&(df['HTA_sr']==0)&(df['DM_sr']==0)&(df['Obesity']==0)),
        ("Edad ≥60 sin HTA/DM/obesidad",
         (df['Age']>=60)&(df['HTA_sr']==0)&(df['DM_sr']==0)&(df['Obesity']==0)),
        ("DM + Obesidad",(df['DM_sr']==1)&(df['Obesity']==1)),
        ("HTA + Obesidad",(df['HTA_sr']==1)&(df['Obesity']==1)),
        ("HTA + DM",(df['HTA_sr']==1)&(df['DM_sr']==1)),
        ("HTA + DM + Obesidad",
         (df['HTA_sr']==1)&(df['DM_sr']==1)&(df['Obesity']==1)),
        ("Edad ≥60 + HTA + DM",
         (df['Age']>=60)&(df['HTA_sr']==1)&(df['DM_sr']==1)),
        ("Antec familiar ERC + Proteinuria+",
         (df['FamHx_ERC']==1)&(df['prot_pos']==1)),
    ]
    # AINEs si está disponible
    # Nota: eGFR<60 NO se usa como criterio para definir subgrupos aquí,
    # porque CKD60 es el desenlace. Usarlo en la máscara produciría una
    # positividad artificialmente igual a 100%.
    if df['AINEs_diario_modelo'].notna().any():
        combos.append(("AINEs diarios",
                       (df['AINEs_diario_modelo'] == 1)))
        combos.append(("Edad ≥60 + AINEs diarios",
                       (df['Age'] >= 60) & (df['AINEs_diario_modelo'] == 1)))
    rows4=[]
    for lbl,mask in combos:
        sv=df[mask.fillna(False)].dropna(subset=['CKD60'])
        n=int((sv['CKD60']==1).sum()); N=int(len(sv))
        if N>=5:
            p,lo,hi=wilson(n,N)
            rows4.append({'Combinación':lbl,'N':N,'n ERC':n,'Prev':p,'lo':lo,'hi':hi})
    if rows4:
        cdf2=pd.DataFrame(rows4).sort_values('Prev')
        pglobal=wilson(int((df['CKD60']==1).sum()),int(df['CKD60'].notna().sum()))[0]
        fig=go.Figure(go.Bar(
            y=cdf2['Combinación'], x=cdf2['Prev'], orientation='h',
            error_x=dict(type='data',array=cdf2['hi']-cdf2['Prev'],
                         arrayminus=cdf2['Prev']-cdf2['lo']),
            marker_color=C_LIGHT))
        anns_c=bar_annotations(cdf2['Combinación'].tolist(), cdf2['Prev'].tolist(),
            cdf2['hi'].tolist(),
            [f"{p:.1f}% (n={n})" for p,n in zip(cdf2['Prev'],cdf2['N'])],
            is_horizontal=True)
        global_ann = []

        xmax = cdf2['hi'].max() * 1.5
        if pd.notna(pglobal):
            xmax = max(xmax, pglobal * 1.35)

            fig.add_vline(
                x=pglobal,
                line_dash='dash',
                line_color='black'
            )

            global_ann.append(dict(
                x=pglobal,
                y=1.08,
                xref='x',
                yref='paper',
                text=f"Global {pglobal:.1f}%",
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=11, color='black'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=0.5,
                borderpad=3
            ))

        fig.update_layout(
            xaxis_title='Positividad (%)',
            xaxis=dict(range=[0, xmax]),
            annotations=anns_c + global_ann,
            height=430,
            margin=dict(l=10, r=160, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 5 — FACTORES DE RIESGO
# ═══════════════════════════════════════════════════════════
with tab_risk:
    st.header("Factores de riesgo")
    st.caption(
        "Esta sección estima razones de positividad (RP) para ERC presuntiva definida como eGFR CKD-EPI 2021 <60. "
        "En estudios transversales, la RP es más interpretable que un OR cuando la positividad no es rara."
    )

    df = add_rural_binary(df)

    st.markdown("### Asociación entre factores clínicos basales y ERC")

    rf_list = [
        ('HTA_sr','HTA conocida (autoreporte)'),
        ('DM_sr','DM conocida (autoreporte)'),
        ('HTA_meas','HTA medida (PA≥130/80)'),
        ('Glc200','Glicemia >200 mg/dL'),
        ('Obesity','Obesidad (cualquier clase)'),
        ('Litiasis_global','Litiasis previa (global)'),
        ('ITU_global','ITU recurrente / pielonefritis (global)'),
        ('FamHx_ERC','Antec familiar ERC (combinado)'),
        ('prot_pos','Proteinuria (+)'),
        ('blood_pos','Hematuria (+)'),
        ('Tabaco_global','Tabaquismo (global)'),
        ('OH_global','Consumo OH problemático/global'),
        ('PPOO_n','Pueblo originario'),
        ('Rural_bin','Ruralidad ≥30%'),
        ('AINEs_diario_modelo','AINEs uso diario (Planilla 03, canónico)'),
        ('Monorreno','Monorreno / asimetría renal'),
        ('Cardio','Cardiopatía / cardiovascular (global)'),
        ('Tiroides','Patología tiroidea (global)'),
        ('DLP_dg','DLP / hipertrigliceridemia (global)'),
    ]

    # Variables estandarizadas basales/exploratorias. Se excluyen Eval_, Accion_, Acción_, Pesquisa_
    # y flags de revisión/dato faltante porque no representan exposición basal.
    std_groups = {
        'Antecedentes personales / diagnósticos': ('antecedentes',),
        'Antecedentes familiares': ('familiares',),
        'AINEs estandarizados': ('aines',),
    }

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        source_sel = st.selectbox(
            "Conjunto de variables",
            ['Factores clínicos predefinidos'] + list(std_groups.keys())
        )
    with c2:
        ajuste_sel = st.selectbox(
            "Tipo de estimación",
            ['Crudo', 'Ajustado por edad + sexo',
             'Ajustado por edad + sexo + PPOO',
             'Ajustado por edad + sexo + PPOO + ruralidad']
        )
    with c3:
        max_vars = st.number_input("Máximo de variables a mostrar", min_value=5, max_value=60, value=25, step=5)

    ajuste_map = {
        'Crudo': [],
        'Ajustado por edad + sexo': ['Age', 'Sex_M'],
        'Ajustado por edad + sexo + PPOO': ['Age', 'Sex_M', 'PPOO_n'],
        'Ajustado por edad + sexo + PPOO + ruralidad': ['Age', 'Sex_M', 'PPOO_n', 'Rural_bin'],
    }
    covars = ajuste_map[ajuste_sel]

    if source_sel == 'Factores clínicos predefinidos':
        vars_to_model = [(v, l) for v, l in rf_list if v in df.columns]
    else:
        std_cols = standardized_risk_columns(df, include_groups=std_groups[source_sel])
        vars_to_model = [(c, humanize_varname(c)) for c in std_cols]

    pr_df = prevalence_ratio_table(df, vars_to_model, outcome='CKD60', covariates=covars)

    if len(pr_df):
        pr_df = pr_df.sort_values('RP', ascending=False).head(int(max_vars)).copy()
        fig = plot_pr_forest(pr_df, title_x=f"Razón de positividad — {ajuste_sel.lower()} (escala log)")
        st.plotly_chart(fig, use_container_width=True)

        tbl = pr_df[['Factor','Variable','N modelo','N expuestos','N no expuestos',
                     'n ERC expuestos','n ERC no exp.','%ERC expuestos','%ERC no exp.',
                     'RP','IC inf','IC sup','p','Método','Ajuste']].copy()
        for c in ['%ERC expuestos','%ERC no exp.','RP','IC inf','IC sup','p']:
            tbl[c] = tbl[c].round(3)
        st.dataframe(style_sig(tbl.rename(columns={'RP':'RP/RR'})), use_container_width=True, hide_index=True)

        if ajuste_sel == 'Crudo':
            st.caption(
                "Estimación cruda: positividad de ERC en expuestos dividida por positividad de ERC en no expuestos. "
                "El IC95% usa aproximación logarítmica."
            )
        else:
            st.caption(
                "Estimación ajustada: modelo de Poisson con enlace log y varianza robusta HC3. "
                "Las variables usadas como ajuste se omiten automáticamente si coinciden con la exposición evaluada."
            )
    else:
        st.info(
            "No hay variables con frecuencia suficiente para este conjunto y nivel de ajuste. "
            "La app exige al menos 5 expuestos, 5 no expuestos y casos de ERC en ambos grupos."
        )

    st.markdown("---")
    st.markdown("### Variables estandarizadas excluidas del análisis de riesgo")
    st.caption(
        "Las columnas derivadas de evaluación médica, acciones clínicas o pesquisa no se interpretan como factores de riesgo. "
        "Además, se omiten flags auxiliares como revisar manual, dato faltante, sin información o antecedentes no especificados. "
        "Estas variables se reservan para análisis de cascada clínica, calidad de datos o conducta posterior."
    )
    excluded_prefixes = ('Eval_', 'Accion_', 'Acción_', 'Pesquisa_')
    candidate_prefixes = ('OtraCondicion_', 'AF_', 'Dg_', 'AINEs_', 'Eval_', 'Accion_', 'Acción_', 'Pesquisa_')
    excluded_cols = [c for c in df.columns if c.startswith(excluded_prefixes) or (c.startswith(candidate_prefixes) and is_nonanalytic_indicator_name(c))]
    excluded_cols = sorted(set(excluded_cols))
    if excluded_cols:
        excl_tbl = pd.DataFrame({
            'Variable excluida': excluded_cols,
            'Etiqueta': [humanize_varname(c) for c in excluded_cols],
            'Motivo': ['Evaluación/acción/pesquisa' if c.startswith(excluded_prefixes) else 'Flag auxiliar / revisión / dato faltante' for c in excluded_cols],
            'N válidos': [int(df[c].notna().sum()) for c in excluded_cols],
            'N positivos': [int(pd.to_numeric(df[c], errors='coerce').sum(skipna=True)) if is_binary_series(df[c]) else np.nan for c in excluded_cols],
        })
        st.dataframe(excl_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No se detectaron columnas excluidas en los datos actualmente cargados.")

    # Subdiagnóstico
    st.markdown("---")
    st.markdown("### Brechas de diagnóstico (subdiagnóstico)")
    hta_d=int((df['HTA_meas']==1).sum())
    hta_c=int(((df['HTA_meas']==1)&(df['HTA_sr']==1)).sum())
    dm_d=int((df['Glc200']==1).sum())
    dm_c=int(((df['Glc200']==1)&(df['DM_sr']==1)).sum())
    ckd_c_sub=df[df['CKD60']==1]
    tto_disp=ckd_c_sub[ckd_c_sub['TTO_ERC_b'].notna()]
    n_tto=len(tto_disp); n_kno=int((tto_disp['TTO_ERC_b']==1).sum())
    tbl_sub=pd.DataFrame([
        ['HTA (PA≥130/80)',hta_d,hta_c,hta_d-hta_c,
         f"{(hta_d-hta_c)/hta_d*100:.1f}%" if hta_d else "—"],
        ['DM (HGT>200)',dm_d,dm_c,dm_d-dm_c,
         f"{(dm_d-dm_c)/dm_d*100:.1f}%" if dm_d else "—"],
        [f'ERC eGFR<60 (sobre n={n_tto} con TTO_ERC disponible)',
         n_tto,n_kno,n_tto-n_kno,
         f"{(n_tto-n_kno)/n_tto*100:.1f}%" if n_tto else "—"],
    ], columns=['Condición','N detectados','Conocidos','Nuevos','% nuevos'])
    st.dataframe(tbl_sub, use_container_width=True, hide_index=True)
    st.caption("El campo TTO_ERC tiene 84% de faltantes entre casos eGFR<60; la cifra defendible es sobre los casos con dato disponible.")


# ═══════════════════════════════════════════════════════════
# TAB 6 — SCORE
# ═══════════════════════════════════════════════════════════
with tab_score:
    st.header("Propuesta de Score clínico de riesgo para tamizaje")
    st.markdown("""
    **Score v4 — 4 variables | rango 0–8 puntos**

    | Variable | Puntos |
    |---|---|
    | Edad <45 / 45–59 / 60–74 / ≥75 | 0 / 1 / 2 / 4 |
    | Proteinuria (+) | 2 |
    | Obesidad (cualquier clase) | 1 |
    | Antecedente familiar ERC | 1 |

    *Score de priorización, no de diagnóstico. Pendiente de validación externa.*
    """)

    val=df.dropna(subset=['SCORE','CKD60']).copy()
    val['ckd']=val['CKD60'].fillna(0).astype(int)
    if len(val)<50:
        st.warning(f"Muestra insuficiente (N={len(val)}).")
    else:
        c1,c2=st.columns(2)
        with c1:
            grp_s=val.groupby('SCORE').agg(n=('ckd','size'),nckd=('ckd','sum'))
            grp_s['prev']=grp_s['nckd']/grp_s['n']*100
            fig=make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(x=grp_s.index,y=grp_s['n'],
                name='N pac.',marker_color=C_LGRAY,opacity=0.8),secondary_y=False)
            fig.add_trace(go.Scatter(x=grp_s.index,y=grp_s['prev'],
                name='Prev ERC %',mode='lines+markers+text',
                line=dict(color=C_ACC,width=3),marker=dict(size=10),
                text=[f"{p:.0f}%" for p in grp_s['prev']],
                textposition='top center'),secondary_y=True)
            fig.update_xaxes(title_text='Puntaje')
            fig.update_yaxes(title_text='N pacientes',secondary_y=False)
            fig.update_yaxes(title_text='Positividad (%)',secondary_y=True,range=[0,100])
            fig.update_layout(height=400,margin=dict(l=10,r=10,t=30,b=20),
                              legend=dict(orientation='h',y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            try:
                auc=roc_auc_score(val['ckd'],val['SCORE'])
                fpr,tpr,_=roc_curve(val['ckd'],val['SCORE'])
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',
                    line=dict(color=C_MAIN,width=2.5),name=f'AUC={auc:.3f}'))
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',
                    line=dict(color=C_GRAY,dash='dash'),name='No discrim.',showlegend=True))
                fig.update_layout(xaxis_title='1-Especificidad',
                    yaxis_title='Sensibilidad',height=400,
                    margin=dict(l=10,r=10,t=30,b=20),legend=dict(x=0.55,y=0.1))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))

        # Estratos
        def estr(s):
            if s<=0: return 'Bajo (0)'
            if s<=2: return 'Intermedio (1-2)'
            if s<=4: return 'Alto (3-4)'
            return 'Muy alto (≥5)'
        val['estr']=val['SCORE'].apply(estr)
        estr_order=['Bajo (0)','Intermedio (1-2)','Alto (3-4)','Muy alto (≥5)']
        e_rows=[]
        for e in estr_order:
            sv=val[val['estr']==e]
            n=int(sv['ckd'].sum()); N=int(len(sv))
            p,lo,hi=wilson(n,N)
            e_rows.append({'Estrato':e,'N':N,'n ERC':n,
                           'Positividad %':f"{p:.1f}%",
                           'IC95%':f"{lo:.1f}–{hi:.1f}"})
        st.dataframe(pd.DataFrame(e_rows), use_container_width=True, hide_index=True)

        # Tabla de cortes
        st.markdown("### Características operativas por corte")
        cut_rows=[]
        for c in sorted(val['SCORE'].dropna().unique()):
            pred=(val['SCORE']>=c).astype(int)
            tp=int(((pred==1)&(val['ckd']==1)).sum())
            fn=int(((pred==0)&(val['ckd']==1)).sum())
            fp=int(((pred==1)&(val['ckd']==0)).sum())
            tn=int(((pred==0)&(val['ckd']==0)).sum())
            sens=tp/(tp+fn) if tp+fn else 0
            esp=tn/(tn+fp) if tn+fp else 0
            vpp=tp/(tp+fp) if tp+fp else 0
            cut_rows.append({'Corte':f'≥{int(c)}','Sens':round(sens,2),
                'Esp':round(esp,2),'VPP':round(vpp,2),
                '% priorizado':round((tp+fp)/(tp+fp+fn+tn)*100,1)})
        st.dataframe(pd.DataFrame(cut_rows), use_container_width=True, hide_index=True)

        # Calculadora
        st.markdown("### 🧮 Calculadora individual")
        c1,c2,c3=st.columns(3)
        with c1:
            i_age=st.slider("Edad",18,100,55)
        with c2:
            i_ob=st.checkbox("Obesidad (cualquier clase)")
            i_fam=st.checkbox("Antec familiar ERC")
        with c3:
            i_prot=st.checkbox("Proteinuria (+)")
        def apt(a):
            return 0 if a<45 else (1 if a<60 else (2 if a<75 else 4))
        sc_u=apt(i_age)+(2 if i_prot else 0)+(1 if i_ob else 0)+(1 if i_fam else 0)
        bp=val[val['SCORE']==sc_u]
        prev_u=(bp['ckd']==1).sum()/len(bp)*100 if len(bp)>0 else np.nan
        c1,c2,c3=st.columns(3)
        c1.metric("Score total",sc_u)
        c2.metric("Prev ERC en este score",
                  f"{prev_u:.1f}%" if pd.notna(prev_u) else "—",
                  help=f"n={len(bp)} con score={sc_u}")
        nivel=('Bajo' if sc_u<=0 else 'Intermedio' if sc_u<=2
               else 'Alto' if sc_u<=4 else 'Muy alto')
        c3.metric("Categoría",nivel)


# ═══════════════════════════════════════════════════════════
# TAB 7 — CONCORDANCIA
# ═══════════════════════════════════════════════════════════
with tab_concord:
    st.header("Concordancia eGFR 1ª vs 2ª medición (CKD-EPI 2021)")

    sub_c=df.dropna(subset=['eGFR','eGFR_2nd']).copy()
    if len(sub_c)<5:
        st.warning(f"Solo {len(sub_c)} registros con segunda toma.")
    else:
        d=sub_c['eGFR_2nd']-sub_c['eGFR']
        bias=d.mean(); sd_d=d.std()
        lo_ba=bias-1.96*sd_d; hi_ba=bias+1.96*sd_d
        rp=pearsonr(sub_c['eGFR'],sub_c['eGFR_2nd'])
        rs=spearmanr(sub_c['eGFR'],sub_c['eGFR_2nd'])
        mx,my=sub_c['eGFR'].mean(),sub_c['eGFR_2nd'].mean()
        sx,sy=sub_c['eGFR'].std(ddof=0),sub_c['eGFR_2nd'].std(ddof=0)
        cov_v=((sub_c['eGFR']-mx)*(sub_c['eGFR_2nd']-my)).mean()
        ccc=2*cov_v/(sx**2+sy**2+(mx-my)**2) if (sx**2+sy**2+(mx-my)**2)>0 else np.nan

        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("N pares",len(sub_c))
        c2.metric("Bias (Δ)",f"{bias:+.2f}")
        c3.metric("Pearson r",f"{rp[0]:.3f}")
        c4.metric("Spearman ρ",f"{rs[0]:.3f}")
        c5.metric("Lin's CCC",f"{ccc:.3f}" if pd.notna(ccc) else "—",
                  help="<0.90 = pobre concordancia")

        col1,col2=st.columns(2)
        with col1:
            st.markdown("**Scatter — línea de identidad**")
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=sub_c['eGFR'],y=sub_c['eGFR_2nd'],
                mode='markers',
                marker=dict(color=C_TEAL,size=8,line=dict(color='white',width=1)),
                hovertemplate='1ª: %{x:.0f}<br>2ª: %{y:.0f}<extra></extra>'))
            mn=min(sub_c['eGFR'].min(),sub_c['eGFR_2nd'].min())
            mx_=max(sub_c['eGFR'].max(),sub_c['eGFR_2nd'].max())
            fig.add_trace(go.Scatter(x=[mn,mx_],y=[mn,mx_],mode='lines',
                line=dict(color=C_GRAY,dash='dash'),showlegend=False))
            fig.add_hline(y=60,line_color=C_ACC,line_dash='dot')
            fig.add_vline(x=60,line_color=C_ACC,line_dash='dot')
            fig.update_layout(xaxis_title='eGFR 1ª toma (CKD-EPI 2021)',
                yaxis_title='eGFR 2ª toma',height=400,margin=dict(l=10,r=10,t=10,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Bland-Altman**")
            mean_v=(sub_c['eGFR']+sub_c['eGFR_2nd'])/2
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=mean_v,y=d,mode='markers',
                marker=dict(color=C_TEAL,size=8,line=dict(color='white',width=1)),
                hovertemplate='Media: %{x:.0f}<br>Δ: %{y:.1f}<extra></extra>'))
            fig.add_hline(y=bias,line_color='black',
                          annotation_text=f'Bias={bias:.1f}')
            fig.add_hline(y=hi_ba,line_color=C_ACC,line_dash='dash',
                          annotation_text=f'+1.96DE={hi_ba:.1f}')
            fig.add_hline(y=lo_ba,line_color=C_ACC,line_dash='dash',
                          annotation_text=f'-1.96DE={lo_ba:.1f}')
            fig.update_layout(xaxis_title='Promedio (1ª+2ª)/2',
                yaxis_title='Δ (2ª-1ª)',height=400,margin=dict(l=10,r=10,t=10,b=20))
            st.plotly_chart(fig, use_container_width=True)

        sub_c['b1']=(sub_c['eGFR']<60).fillna(False).astype(int)
        sub_c['b2']=(sub_c['eGFR_2nd']<60).fillna(False).astype(int)
        tab_reclasif=pd.crosstab(
            sub_c['b1'].map({0:'≥60 (1ª)',1:'<60 (1ª)'}),
            sub_c['b2'].map({0:'≥60 (2ª)',1:'<60 (2ª)'}),margins=True)
        st.markdown("### Reclasificación binaria (eGFR<60)")
        st.dataframe(tab_reclasif, use_container_width=True)
        g1_k=sub_c['eGFR'].apply(kdigo_g); g2_k=sub_c['eGFR_2nd'].apply(kdigo_g)
        same_k=int((g1_k==g2_k).sum())
        st.caption(f"Misma categoría KDIGO G: {same_k}/{len(sub_c)} ({same_k/len(sub_c)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════
# TAB 8 — LITERATURA (Comparación con análisis previamente publicados)
# ═══════════════════════════════════════════════════════════
with tab_lit:
    st.header("Análisis realizados previamente en trabajos publicados")
    st.caption("Zúñiga 2011 · Walbaum 2020 · Meneses 2023 · Poblete 2024")

    sub_t = st.radio("Selecciona análisis",
        ["ERC oculta","Correlación eGFR-Edad","U invertida KDIGO",
         "Etiología presuntiva","Interacción DM × HTA",
         "AINEs y función renal","Subdiagnóstico en HTA/DM (PSCV proxy)"],
        horizontal=True)

    if sub_t == "ERC oculta":
        st.markdown("### ERC oculta — Zúñiga 2011")
        st.info("Definición: eGFR<60 con creatinina capilar ≤1,0 mg/dL. Pacientes con daño renal oculto bajo una creatinina aparentemente normal — riesgo de prescripción de nefrotóxicos.")
        ckd_cases=df[df['CKD60']==1]
        eo_n=int((ckd_cases['Creatinine_1st']<=1.0).sum())
        eo_N=int(ckd_cases['Creatinine_1st'].notna().sum())
        p,lo,hi=wilson(eo_n,eo_N)
        c1,c2,c3=st.columns(3)
        c1.metric("ERC oculta",f"{eo_n} casos",f"{p:.1f}% de eGFR<60")
        c2.metric("IC95%",f"{lo:.1f}–{hi:.1f}%")
        c3.metric("Referencia Zúñiga 2011","26,8%",
                  "MDRD-4, APS Concepción — diferencia por ecuación")
        # Por sexo y edad
        rows_eo=[]
        for sex in ['Mujer','Hombre']:
            sv=ckd_cases[ckd_cases['Sex_lbl']==sex]
            n=int((sv['Creatinine_1st']<=1.0).sum()); N=int(sv['Creatinine_1st'].notna().sum())
            p2,lo2,hi2=wilson(n,N)
            rows_eo.append({'Subgrupo':sex,'n ERC oculta':n,'N eGFR<60':N,
                            'Positividad':f"{p2:.1f}%",'IC95%':f"{lo2:.1f}–{hi2:.1f}"})
        for g in ['<30','30-44','45-59','60-74','≥75']:
            sv=ckd_cases[ckd_cases['Age_grp']==g]
            n=int((sv['Creatinine_1st']<=1.0).sum()); N=int(sv['Creatinine_1st'].notna().sum())
            p2,lo2,hi2=wilson(n,N)
            rows_eo.append({'Subgrupo':f"Edad {g}",'n ERC oculta':n,'N eGFR<60':N,
                            'Positividad':f"{p2:.1f}%",'IC95%':f"{lo2:.1f}–{hi2:.1f}"})
        st.dataframe(pd.DataFrame(rows_eo), use_container_width=True, hide_index=True)

    elif sub_t == "Correlación eGFR-Edad":
        st.markdown("### Correlación continua eGFR–Edad — Zúñiga 2011")
        sub_cr=df.dropna(subset=['eGFR','Age'])
        rp_v,pp_v=pearsonr(sub_cr['Age'],sub_cr['eGFR'])
        rs_v,ps_v=spearmanr(sub_cr['Age'],sub_cr['eGFR'])
        c1,c2,c3=st.columns(3)
        c1.metric("Pearson r",f"{rp_v:.3f}","p<0.001")
        c2.metric("Spearman ρ",f"{rs_v:.3f}")
        c3.metric("Referencia Zúñiga 2011","r = –0,54","MDRD-4, APS Concepción")
        samp=sub_cr.sample(min(800,len(sub_cr)),random_state=42)
        lr=linregress(samp['Age'],samp['eGFR'])
        x_r=np.linspace(samp['Age'].min(),samp['Age'].max(),100)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=samp['Age'],y=samp['eGFR'],mode='markers',
            marker=dict(color=samp['eGFR'],colorscale='RdYlGn',cmin=10,cmax=130,
                        size=5,opacity=0.5,showscale=True,colorbar=dict(title='eGFR')),
            hovertemplate='Edad %{x}<br>eGFR %{y:.0f}<extra></extra>',showlegend=False))
        fig.add_trace(go.Scatter(x=x_r,y=lr.slope*x_r+lr.intercept,mode='lines',
            line=dict(color=C_MAIN,width=2.5),name=f"r = {rp_v:.3f}"))
        fig.add_hline(y=60,line_color=C_ACC,line_dash='dash',
                      annotation_text='eGFR 60')
        fig.update_layout(xaxis_title='Edad (años)',
            yaxis_title='eGFR CKD-EPI 2021 (mL/min/1.73m²)',
            height=450,margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Comparación metodológica directa con Zúñiga 2011: Edad × MDRD-4")
        st.caption("Este bloque recalcula eGFR con MDRD-4 clásico solo para comparar la correlación continua edad–eGFR con Zúñiga 2011. No cambia la definición principal del dashboard.")

        sub_mdrd = df.dropna(subset=['Age','eGFR_MDRD4_Zuniga','eGFR'])
        if len(sub_mdrd) >= 5:
            rp_m, pp_m = pearsonr(sub_mdrd['Age'], sub_mdrd['eGFR_MDRD4_Zuniga'])
            rs_m, ps_m = spearmanr(sub_mdrd['Age'], sub_mdrd['eGFR_MDRD4_Zuniga'])
            lr_m = linregress(sub_mdrd['Age'], sub_mdrd['eGFR_MDRD4_Zuniga'])
            lr_epi = linregress(sub_mdrd['Age'], sub_mdrd['eGFR'])

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Pearson r MDRD-4", f"{rp_m:.3f}", "Zúñiga: r = –0,54")
            c2.metric("Spearman ρ MDRD-4", f"{rs_m:.3f}")
            c3.metric("Pendiente MDRD-4", f"{lr_m.slope:.2f}", "mL/min/1.73m² por año")
            c4.metric("Diferencia vs CKD-EPI", f"{rp_m - rp_v:+.3f}", "Δ Pearson r")

            samp_m = sub_mdrd.sample(min(800, len(sub_mdrd)), random_state=42)
            x_r_m = np.linspace(samp_m['Age'].min(), samp_m['Age'].max(), 100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=samp_m['Age'], y=samp_m['eGFR_MDRD4_Zuniga'], mode='markers',
                marker=dict(color=C_TEAL, size=5, opacity=0.45),
                name='MDRD-4',
                hovertemplate='Edad %{x}<br>MDRD-4 %{y:.0f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=x_r_m, y=lr_m.slope*x_r_m + lr_m.intercept, mode='lines',
                line=dict(color=C_MAIN, width=2.8),
                name=f'MDRD-4: r={rp_m:.3f}'
            ))
            fig.add_trace(go.Scatter(
                x=x_r_m, y=lr_epi.slope*x_r_m + lr_epi.intercept, mode='lines',
                line=dict(color=C_ACC, width=2.2, dash='dash'),
                name=f'CKD-EPI 2021: r={rp_v:.3f}'
            ))
            fig.add_hline(y=60, line_color=C_GRAY, line_dash='dot', annotation_text='eGFR 60')
            fig.update_layout(
                xaxis_title='Edad (años)',
                yaxis_title='eGFR (mL/min/1.73m²)',
                height=450, margin=dict(l=10,r=10,t=20,b=20),
                legend=dict(orientation='h', y=1.08)
            )
            st.plotly_chart(fig, use_container_width=True)

            comp = pd.DataFrame([
                {'Ecuación':'CKD-EPI 2021', 'N':len(sub_mdrd), 'Pearson r':rp_v, 'Spearman ρ':rs_v, 'Pendiente/año':lr_epi.slope},
                {'Ecuación':'MDRD-4 tipo Zúñiga', 'N':len(sub_mdrd), 'Pearson r':rp_m, 'Spearman ρ':rs_m, 'Pendiente/año':lr_m.slope},
                {'Ecuación':'Zúñiga 2011 reportado', 'N':'27.894', 'Pearson r':-0.54, 'Spearman ρ':np.nan, 'Pendiente/año':np.nan},
            ])
            st.dataframe(comp, use_container_width=True, hide_index=True)
            st.info(
                "Lectura: esta comparación usa la misma familia de ecuación reportada por Zúñiga "
                "—MDRD-4 con factor 0,742 en mujeres— para que la correlación edad–eGFR sea metodológicamente más comparable. "
                "La línea CKD-EPI 2021 se conserva como referencia interna del dashboard."
            )
        else:
            st.warning("No hay datos suficientes para calcular la comparación MDRD-4.")

    elif sub_t == "U invertida KDIGO":
        st.markdown("### 'U invertida': edad media por estadio KDIGO G — Walbaum 2020")
        st.info("La edad media sube al avanzar el estadio hasta G3b, luego cae en G4–G5 por mortalidad selectiva: los adultos mayores con ERC avanzada no llegan al tamizaje.")
        ui_rows=[]
        for g in ['G1','G2','G3a','G3b','G4','G5']:
            sv=df[df['KDIGO_G']==g]['Age'].dropna()
            if len(sv)<3: continue
            se=sv.sem()
            ui_rows.append({'Estadio':g,'N':len(sv),'Edad media':round(sv.mean(),1),
                            'IC inf':round(sv.mean()-1.96*se,1),
                            'IC sup':round(sv.mean()+1.96*se,1),'DE':round(sv.std(),1)})
        if ui_rows:
            udf=pd.DataFrame(ui_rows)
            fig=go.Figure()
            fig.add_trace(go.Scatter(
                x=udf['Estadio'],y=udf['Edad media'],mode='lines+markers+text',
                error_y=dict(type='data',
                    array=udf['IC sup']-udf['Edad media'],
                    arrayminus=udf['Edad media']-udf['IC inf']),
                line=dict(color=C_MAIN,width=2.5),
                marker=dict(color=[KDIGO_COLORS[g] for g in udf['Estadio']],size=14,
                            line=dict(color='white',width=2)),
                text=[f"{m:.1f}" for m in udf['Edad media']],
                textposition='top center'))
            fig.update_layout(xaxis_title='Estadio KDIGO G',
                yaxis_title='Edad media (años)',
                height=420,margin=dict(l=10,r=10,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(udf, use_container_width=True, hide_index=True)

    elif sub_t == "Etiología presuntiva":
        st.markdown("### Etiología presuntiva de la ERC detectada — Poblete 2024")
        st.info("Clasificación operacional basada en antecedentes clínicos. Referencia nacional (HDC 2024): nefropatía diabética 36,3%, nefroesclerosis HTA 14,8%, causa desconocida 5,5%.")
        ckd_e=df[df['CKD60']==1].copy()
        ckd_e['DM_any']=(ckd_e['DM_sr']==1)|(ckd_e['Glc200']==1)
        ckd_e['HTA_any']=(ckd_e['HTA_sr']==1)|(ckd_e['HTA_meas']==1)
        def etio(r):
            dm=r['DM_any']; hta=r['HTA_any']
            if dm and hta: return 'DM + HTA'
            if dm: return 'Solo DM'
            if hta: return 'Solo HTA'
            return 'Origen incierto'
        ckd_e['etio']=ckd_e.apply(etio,axis=1)
        ec=ckd_e['etio'].value_counts()
        fig=go.Figure(go.Pie(
            labels=ec.index,values=ec.values,
            marker=dict(colors=[C_ACC,'#F4E04D','#F08C5A',C_GRAY]),
            hole=0.4,textinfo='label+percent+value',textposition='outside'))
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        rows_et=[]
        for e in ['DM + HTA','Solo HTA','Solo DM','Origen incierto']:
            n=int((ckd_e['etio']==e).sum()); N=len(ckd_e)
            p,lo,hi=wilson(n,N)
            rows_et.append({'Etiología presuntiva':e,'n':n,'%':f"{p:.1f}%",
                            'IC95%':f"{lo:.1f}–{hi:.1f}"})
        st.dataframe(pd.DataFrame(rows_et), use_container_width=True, hide_index=True)

    elif sub_t == "Interacción DM × HTA":
        st.markdown("### Interacción DM × HTA — Walbaum 2020")
        st.info("Walbaum 2020 reportó OR 2,30 para la combinación DM+HTA vs DM sola sobre albuminuria.")
        combis=[('Ni HTA ni DM',(df['HTA_sr']==0)&(df['DM_sr']==0)),
                ('Solo HTA',(df['HTA_sr']==1)&(df['DM_sr']==0)),
                ('Solo DM',(df['HTA_sr']==0)&(df['DM_sr']==1)),
                ('HTA + DM',(df['HTA_sr']==1)&(df['DM_sr']==1))]
        rows_int=[]
        for lbl,mask in combis:
            sv=df[mask.fillna(False)].dropna(subset=['CKD60'])
            n=int((sv['CKD60']==1).sum()); N=int(len(sv))
            p,lo,hi=wilson(n,N)
            rows_int.append({'Combinación':lbl,'N':N,'n ERC':n,
                             'Prev %':round(p,1),'lo':lo,'hi':hi})
        rdf_int=pd.DataFrame(rows_int)
        fig=go.Figure()
        colors_int=[C_OK,'#FFDD00',C_TEAL,C_ACC]
        fig.add_trace(go.Bar(x=rdf_int['Combinación'],y=rdf_int['Prev %'],
            error_y=dict(type='data',array=rdf_int['hi']-rdf_int['Prev %'],
                         arrayminus=rdf_int['Prev %']-rdf_int['lo']),
            marker_color=colors_int,
            text=[f"{p:.1f}%<br>n={N}" for p,N in zip(rdf_int['Prev %'],rdf_int['N'])],
            textposition='outside'))
        fig.update_layout(yaxis_title='Prevalencia eGFR<60 (%)',
            height=400,margin=dict(l=10,r=10,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)
        try:
            dat_int=df[['CKD60','Age','Sex_M','PPOO_n','Obesity','HTA_sr','DM_sr',
                        'FamHx_ERC','prot_pos']].dropna().copy()
            dat_int['ckd']=dat_int['CKD60'].fillna(0).astype(int)
            dat_int['htadm']=dat_int['HTA_sr']*dat_int['DM_sr']
            m1=smf.logit('ckd~Age+Sex_M+PPOO_n+Obesity+HTA_sr+DM_sr+FamHx_ERC+prot_pos',
                          data=dat_int).fit(disp=0)
            m2=smf.logit('ckd~Age+Sex_M+PPOO_n+Obesity+HTA_sr+DM_sr+htadm+FamHx_ERC+prot_pos',
                          data=dat_int).fit(disp=0)
            or_int=np.exp(m2.params['htadm'])
            ci_int=(np.exp(m2.conf_int().loc['htadm',0]),
                    np.exp(m2.conf_int().loc['htadm',1]))
            pv_int=m2.pvalues['htadm']
            c1,c2,c3=st.columns(3)
            c1.metric("OR interacción HTA×DM",f"{or_int:.2f}")
            c2.metric("IC95%",f"{ci_int[0]:.2f}–{ci_int[1]:.2f}")
            c3.metric("p",f"{pv_int:.4f}")
            st.caption(f"AIC sin interacción={m1.aic:.1f} | con interacción={m2.aic:.1f}")
        except Exception as e:
            st.warning(f"No se pudo ajustar modelo de interacción: {e}")

    elif sub_t == "AINEs y función renal":
        st.markdown("### AINEs y función renal — orientación nefróloga")
        st.info("Consumo de AINEs en pacientes con ERC no diagnosticada = nefrotoxicidad evitable. Relevancia clínica directa para prescripción segura en terreno.")
        if df['AINEs_diario_modelo'].notna().sum() < 20:
            # Use raw Consumo AINEs
            if 'Consumo AINEs' in df.columns:
                cats=df['Consumo AINEs'].value_counts()
                cat_order=['Nunca','Cuando sea necesario','Unas pocas al mes',
                           '1-4 al día','5-10 al día']
                cat_order=[c for c in cat_order if c in cats.index]
                aines_prev=[]
                for c in cat_order:
                    sv=df[df['Consumo AINEs']==c].dropna(subset=['CKD60'])
                    n=int((sv['CKD60']==1).sum()); N=int(len(sv))
                    p,lo,hi=wilson(n,N)
                    aines_prev.append({'Categoría':c,'N':N,'n ERC':n,
                                       'Prev %':round(p,1),'lo':lo,'hi':hi})
        else:
            aines_map_r={
                'AINEs_NoUso_Reportado':'No uso reportado',
                'AINEs_Uso_PRN_NoCuantificado':'Uso PRN (sin cuantificar)',
                'AINEs_Uso_OcasionalMensual':'Uso ocasional mensual',
                'AINEs_Uso_Diario':'Uso diario (1-4/día)',
                'AINEs_Uso_Diario_Alto_5a10':'Uso diario alto (5-10/día)',
            }
            aines_prev=[]
            for col,lbl in aines_map_r.items():
                if col not in df.columns: continue
                sv=df[df[col]==1].dropna(subset=['CKD60'])
                if len(sv)<5: continue
                n=int((sv['CKD60']==1).sum()); N=int(len(sv))
                p,lo,hi=wilson(n,N)
                aines_prev.append({'Categoría':lbl,'N':N,'n ERC':n,
                                   'Prev %':round(p,1),'lo':lo,'hi':hi})

        if aines_prev:
            adf=pd.DataFrame(aines_prev)
            cols_a=['#4DAF4A','#FFDD00','#F28F00','#E83800','#9E0000'][:len(adf)]
            fig=go.Figure()
            fig.add_trace(go.Bar(x=adf['Categoría'],y=adf['Prev %'],
                marker_color=cols_a,
                error_y=dict(type='data',array=adf['hi']-adf['Prev %'],
                             arrayminus=adf['Prev %']-adf['lo'])))
            anns_a=bar_annotations(adf['Categoría'].tolist(),adf['Prev %'].tolist(),
                adf['hi'].tolist(),
                [f"{p:.1f}%<br>n={N}" for p,N in zip(adf['Prev %'],adf['N'])])
            pglobal=wilson(int((df['CKD60']==1).sum()),int(df['CKD60'].notna().sum()))[0]
            if pd.notna(pglobal):
                fig.add_hline(y=pglobal,line_dash='dash',line_color='black',
                              annotation_text=f"Global {pglobal:.1f}%")
            fig.update_layout(yaxis_title='Positividad eGFR<60 (%)',
                annotations=anns_a,
                yaxis=dict(range=[0, adf['hi'].max()*1.45 if len(adf) else 30]),
                height=400,margin=dict(l=10,r=10,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(adf, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay datos suficientes de AINEs en la muestra filtrada.")

        # ERC oculta + AINEs
        eo=df[(df['CKD60']==1)&(df['Creatinine_1st']<=1.0)]
        if len(eo)>0:
            uso_eo=int((eo['AINEs_diario_modelo']==1).sum()) if df['AINEs_diario_modelo'].notna().sum()>0 else 0
            st.info(f"**ERC oculta con uso de AINEs:** {uso_eo}/{len(eo)} casos de ERC oculta "
                    f"reportan uso de AINEs — riesgo de nefrotoxicidad en pacientes con "
                    f"creatinina aparentemente normal.")

    elif sub_t == "Subdiagnóstico en HTA/DM (PSCV proxy)":
        st.markdown("### Subdiagnóstico de ERC en proxy PSCV — Zúñiga 2011")
        st.info("Personas con HTA y/o DM autoreportada ≈ candidatos al PSCV. Zúñiga 2011: solo 1,1% de pacientes con ERC dentro del PSCV tenían el diagnóstico de ERC consignado en ficha.")
        pscv=df[(df['HTA_sr']==1)|(df['DM_sr']==1)]
        pscv_ckd=pscv[pscv['CKD60']==1]
        tto=pscv_ckd[pscv_ckd['TTO_ERC_b'].notna()]
        n_tto=len(tto); n_kno=int((tto['TTO_ERC_b']==1).sum())
        c1,c2,c3,c4=st.columns(4)
        c1.metric("N proxy PSCV",f"{len(pscv):,}")
        c2.metric("ERC en PSCV",f"{len(pscv_ckd)}",
                  f"{len(pscv_ckd)/len(pscv)*100:.1f}%")
        c3.metric("Con TTO_ERC disponible",f"{n_tto}")
        c4.metric("% sin diagnóstico previo",
                  f"{(n_tto-n_kno)/n_tto*100:.1f}%" if n_tto else "—",
                  help="Sobre los que tienen dato disponible")

        # Pirámide PSCV vs no-PSCV
        rows_pscv=[]
        for g in ['<30','30-44','45-59','60-74','≥75']:
            for grp_l,grp_m in [('PSCV',pscv),('No PSCV',df[~df.index.isin(pscv.index)])]:
                sv=grp_m[grp_m['Age_grp']==g].dropna(subset=['CKD60'])
                if len(sv)<5: continue
                n=int((sv['CKD60']==1).sum()); N=int(len(sv))
                p,lo,hi=wilson(n,N)
                rows_pscv.append({'Edad':g,'Grupo':grp_l,'Prev':p,'N':N})
        if rows_pscv:
            pdf2=pd.DataFrame(rows_pscv)
            fig=px.bar(pdf2,x='Edad',y='Prev',color='Grupo',barmode='group',
                color_discrete_map={'PSCV':C_ACC,'No PSCV':C_LIGHT},
                category_orders={'Edad':['<30','30-44','45-59','60-74','≥75']},
                hover_data=['N'])
            fig.update_layout(yaxis_title='Positividad eGFR<60 (%)',
                height=400,margin=dict(l=10,r=10,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 9 — DATOS
# ═══════════════════════════════════════════════════════════
with tab_data:
    st.header("Datos filtrados")
    st.caption(f"N seleccionado: **{len(df):,}**")

    default_cols=['ID','Age','Sex_lbl','Community_std','PPOO_lbl','BMI_cat',
                  'BP_Final','eGFR','KDIGO_G','Alb_cat','KDIGO_risk',
                  'CKD60','prot_pos','AINEs_diario_modelo','FamHx_ERC']
    default_cols=[c for c in default_cols if c in df.columns]
    cols_show=st.multiselect("Columnas a mostrar", df.columns.tolist(),
                              default=default_cols)
    if cols_show:
        st.dataframe(df[cols_show], use_container_width=True, height=520)
        csv=df[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar CSV",csv,'ckd_filtrado.csv','text/csv')

st.markdown("---")
st.caption(
    "Dashboard ERC Araucanía v4.0 · eGFR CKD-EPI 2021 · Planillas 01-05 · "
    "Positividad aplican a la muestra de screening, no a la población general."
)
