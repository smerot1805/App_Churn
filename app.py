from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).parent / "data" / "customer_churn.csv"
FEATURES = ["Age", "Num_Sites", "Account_Manager", "Total_Purchase", "Years"]

st.set_page_config(page_title="Signal | Churn", page_icon="S", layout="wide")


@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + ["Churn"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {', '.join(missing)}")

    training_data = data[required].dropna()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(training_data[FEATURES], training_data["Churn"].astype(int))
    return model, training_data["Churn"].mean(), len(training_data)


model, churn_rate, customer_count = train_model()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --ink: #17251f;
        --muted: #65736b;
        --line: #dce4dc;
        --green: #217451;
        --canvas: #fff1f4;
    }
    .stApp {
        background-color: var(--canvas);
        background-image: repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(158, 76, 99, .045) 40px), repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(158, 76, 99, .045) 40px);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    .block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); letter-spacing: 0 !important; }
    .eyebrow, .mono { font-family: 'DM Mono', monospace; letter-spacing: 0 !important; }
    .eyebrow { color: var(--green); font-size: .72rem; text-transform: uppercase; }
    .masthead { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); padding-bottom: 1.3rem; margin-bottom: 2rem; }
    .brand { font: 800 1rem 'Manrope', sans-serif; letter-spacing: 0 !important; }
    .brand-mark { display: inline-grid; place-items: center; width: 28px; height: 28px; margin-right: 9px; background: var(--green); color: white; font: 500 .8rem 'DM Mono', monospace; }
    .masthead-meta { color: var(--muted); font: .7rem 'DM Mono', monospace; text-transform: uppercase; }
    .intro { padding: .25rem 0 1.2rem; }
    .intro h1 { font-size: 2.5rem; line-height: 1.12; margin: .55rem 0 .45rem; }
    .intro p { color: var(--muted); margin: 0; font-size: .98rem; }
    .section-label { color: var(--muted); font: .7rem 'DM Mono', monospace; text-transform: uppercase; border-bottom: 1px solid var(--line); padding-bottom: .7rem; margin: .5rem 0 1.1rem; }
    [data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label { color: var(--ink); font-size: .86rem; font-weight: 600; }
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: rgba(255,255,255,.72); border-color: var(--line); border-radius: 3px; }
    .stButton > button { background: var(--green); color: white; border: 1px solid var(--green); border-radius: 3px; min-height: 3rem; font-weight: 700; transition: background .16s ease, transform .16s ease; }
    .stButton > button:hover { background: #185d40; border-color: #185d40; color: white; transform: translateY(-1px); }
    .result-panel { min-height: 350px; padding: 1.6rem; background: rgba(255,255,255,.78); border: 1px solid var(--line); border-top: 3px solid var(--green); }
    .result-kicker { color: var(--muted); font: .7rem 'DM Mono', monospace; text-transform: uppercase; }
    .result-number { color: var(--ink); font: 800 4.5rem/.98 'Manrope', sans-serif; letter-spacing: 0 !important; margin: 1rem 0 .8rem; }
    .risk-tag { display: inline-block; padding: .38rem .6rem; font: 500 .72rem 'DM Mono', monospace; text-transform: uppercase; border: 1px solid currentColor; }
    .risk-track { width: 100%; height: 7px; margin: 1.5rem 0 .55rem; background: #e9ede8; }
    .risk-fill { height: 7px; }
    .scale { display: flex; justify-content: space-between; color: var(--muted); font: .65rem 'DM Mono', monospace; }
    .result-note { margin-top: 2.1rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .83rem; line-height: 1.5; }
    .baseline { display: flex; justify-content: space-between; gap: 1rem; padding: .85rem 0; border-bottom: 1px solid var(--line); color: var(--muted); font-size: .82rem; }
    .baseline strong { color: var(--ink); font-family: 'DM Mono', monospace; font-weight: 500; }
    .footer { margin-top: 2.5rem; border-top: 1px solid var(--line); padding-top: 1rem; display: flex; justify-content: space-between; gap: 1rem; color: var(--muted); font: .67rem 'DM Mono', monospace; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .masthead-meta { display: none; }
        .intro h1 { font-size: 2rem; }
        .result-number { font-size: 3.7rem; }
        .footer { flex-direction: column; gap: .35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="masthead"><div class="brand"><span class="brand-mark">S</span>SIGNAL</div>'
    '<div class="masthead-meta">Customer intelligence / 01</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="intro"><div class="eyebrow">Rétention · Analyse prédictive</div>'
    '<h1>Risque de départ</h1>'
    '<p>Estimation individuelle à partir des indicateurs du compte.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.markdown('<div class="section-label">Profil du compte</div>', unsafe_allow_html=True)
    with st.form("customer_profile"):
        first_row, second_row = st.columns(2)
        with first_row:
            age = st.number_input("Âge", min_value=22, max_value=65, value=42, step=1)
        with second_row:
            total_purchase = st.number_input(
                "Achats cumulés", min_value=100.0, max_value=18026.01, value=10000.0, step=100.0, format="%.2f"
            )
        first_row, second_row = st.columns(2)
        with first_row:
            num_sites = st.number_input("Nombre de sites", min_value=3, max_value=14, value=8, step=1)
        with second_row:
            years = st.number_input("Ancienneté (années)", min_value=1.0, max_value=9.15, value=5.0, step=0.1, format="%.1f")
        account_manager = st.selectbox("Account manager", ["Non", "Oui"])
        submitted = st.form_submit_button("Calculer le risque", use_container_width=True)

    if submitted:
        profile = pd.DataFrame(
            [[age, num_sites, int(account_manager == "Oui"), total_purchase, years]],
            columns=FEATURES,
        )
        st.session_state["churn_probability"] = float(model.predict_proba(profile)[0, 1])

with right:
    st.markdown('<div class="section-label">Résultat du modèle</div>', unsafe_allow_html=True)
    probability = st.session_state.get("churn_probability")
    if probability is None:
        st.markdown(
            '<div class="result-panel"><div class="result-kicker">Probabilité de churn</div>'
            '<div class="result-number">--.-%</div><div class="result-note">Le score apparaîtra ici après le calcul du profil.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        if probability < 0.20:
            label, color = "Faible", "#217451"
        elif probability < 0.45:
            label, color = "Modéré", "#b66a17"
        else:
            label, color = "Élevé", "#b84339"
        percentage = probability * 100
        st.markdown(
            f'<div class="result-panel" style="border-top-color:{color}">'
            f'<div class="result-kicker">Probabilité de churn</div>'
            f'<div class="result-number">{percentage:.1f}<span style="font-size:2rem">%</span></div>'
            f'<span class="risk-tag" style="color:{color}">{label}</span>'
            f'<div class="risk-track"><div class="risk-fill" style="width:{percentage:.1f}%;background:{color}"></div></div>'
            '<div class="scale"><span>0 %</span><span>100 %</span></div>'
            '<div class="result-note">Estimation statistique basée sur les profils historiques. Ce score indique une tendance, pas une certitude.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="baseline"><span>Taux de churn historique</span><strong>{churn_rate:.1%}</strong></div>'
        f'<div class="baseline"><span>Comptes analysés</span><strong>{customer_count}</strong></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer"><span>MODÈLE · RÉGRESSION LOGISTIQUE</span>'
    '<span>VARIABLES · ÂGE / SITES / MANAGER / ACHATS / ANCIENNETÉ</span></div>',
    unsafe_allow_html=True,
)