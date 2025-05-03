import streamlit as st
import pandas as pd
import numpy as np                           # ← ya lo tenías
# ❌  import plotly.express as px             # ← elimina esta línea

@st.cache_data
def load_data():
    df_symptoms = pd.read_csv(
        "df_symptoms_prob.csv"
    )
    df_prev = pd.read_csv(
        "df_diseases.csv"
    )
    return df_symptoms, df_prev


df_symptoms, df_prevalence = load_data()

def inferir_enfermedad(
    sintomas_usuario,
    df_symptoms,
    df_prevalence,
    *,
    epsilon=1e-9        # evita log(0)
):
    """
    Devuelve:
        - top3_norm  : lista [(enfermedad, P(E|S))] ordenada desc.
        - probs_norm : dict  {enf: P(E|S)}  (posterior normalizado)
        - probs_raw  : dict  {enf: P(E,S)} (joint sin normalizar)
    """
    log_probs = {}
    symptoms_cols = [c for c in df_symptoms.columns if c != "diseases"]

    for _, row in df_symptoms.iterrows():
        enf = row["diseases"]

        # P(E)  (prevalencia)
        prev = df_prevalence.loc[
            df_prevalence["disease"] == enf, "proporcion"
        ].values
        if prev.size == 0:
            continue  # si faltara la prevalencia, ignórala
        log_p = np.log(prev[0] + epsilon)

        # Para cada síntoma de la base (presente o ausente)
        for s in symptoms_cols:
            p_s_e = row[s] if not pd.isna(row[s]) else 0.0

            if s in sintomas_usuario:          # usuario SÍ lo tiene
                log_p += np.log(p_s_e + epsilon)
            else:                              # usuario NO lo tiene
                log_p += np.log(1 - p_s_e + epsilon)

        log_probs[enf] = log_p

    # ---- Paso de normalización ----
    max_log = max(log_probs.values())
    probs_raw = {e: np.exp(lp - max_log)          # joint sin normalizar
                 for e, lp in log_probs.items()}
    total = sum(probs_raw.values())
    probs_norm = {e: p / total for e, p in probs_raw.items()}

    top3_norm = sorted(probs_norm.items(),
                       key=lambda x: x[1],
                       reverse=True)[:3]
    return top3_norm, probs_norm, probs_raw
st.title("🔍 Sistema Experto de Diagnóstico")

with st.sidebar:
    st.header("🩺 Selecciona tus síntomas")
    sintomas_disponibles = [c for c in df_symptoms.columns if c != "diseases"]
    sintomas_usuario = st.multiselect(
        "Síntomas (1-10)",
        options=sintomas_disponibles,
        help="Ctrl/Cmd + clic para elegir varios",
    )
    diagnosticar = st.button("Diagnosticar 🚑")

if diagnosticar:
    n = len(sintomas_usuario)
    if n == 0:
        st.warning("⚠️ Selecciona al menos 1 síntoma.")
        st.stop()
    if n > 10:
        st.warning("⚠️ Máximo 10 síntomas. Quita alguno para continuar.")
        st.stop()

    with st.spinner("Analizando…"):
        top3, probs_norm, probs_raw = inferir_enfermedad(
            sintomas_usuario, df_symptoms, df_prevalence
        )

    st.subheader("🏥 Top 3 enfermedades más probables")
    col1, col2, col3 = st.columns(3)
    for col, (enf, pnorm) in zip([col1, col2, col3], top3):
        col.metric(label=f"🩻 {enf}", value=f"{pnorm:.1%}")

    # ---------- NUEVO bloque: gráfica sin Plotly ----------
    import altair as alt  # ← agrégalo si no lo tienes ya

    df_plot = pd.DataFrame(top3, columns=["Enfermedad", "Probabilidad"])
    df_plot["rank"] = df_plot["Probabilidad"].rank(ascending=False, method="first")
    df_plot = df_plot.sort_values("rank")

    chart = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X("Enfermedad", sort="-y"),
        y="Probabilidad"
    )
    st.altair_chart(chart, use_container_width=True)
       # usa el componente nativo de Streamlit
    # -------------------------------------------------------

    with st.expander("📋 Ver detalles completos"):
        df_detalle = pd.DataFrame(
            {
                "Enfermedad": probs_raw.keys(),
                "P(E,S) sin normalizar": probs_raw.values(),
                "P(E|S) normalizada": [probs_norm[e] for e in probs_raw],
            }
        )
        st.dataframe(
            df_detalle.style.format(
                {
                    "P(E,S) sin normalizar": "{:.3e}",
                    "P(E|S) normalizada": "{:.3%}",
                }
            ),
            use_container_width=True,
        )

st.markdown(
    """
    <hr style="margin-top:2rem;margin-bottom:1rem;">
    <small>
    Este sistema fue realizado como un ejercicio educativo, no sustituye un servicio médico especializado. Chécate, mídete, muévete.
    </small>
    <small> Comienza seleccionando tus sintomas en el panel de la izquierda.
    </small>
    """,
    unsafe_allow_html=True,
)
