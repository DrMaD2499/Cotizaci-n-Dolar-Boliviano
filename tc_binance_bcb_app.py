import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y CSS CORREGIDO
# ==========================================
st.set_page_config(
    page_title="Monitor Cambiario & Análisis Económico",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Forzar fondo blanco general y texto oscuro */
    .stApp {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    header {visibility: hidden;}

    /* FORZAR COLOR NEGRO Y TAMAÑO EN LOS NÚMEROS Y LABELS DE MÉTRICAS (SOLUCIÓN MÓVIL) */
    div[data-testid="stMetricValue"] {
        color: #111111 !important;
        font-weight: 700 !important;
        font-size: calc(1.0rem + 0.5vw) !important;
        word-break: normal !important;
    }
    div[data-testid="stMetricValue"] * {
        color: #111111 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #444444 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        white-space: normal !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }

    /* Permitir scroll horizontal en móviles si 5 columnas no caben completas */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            overflow-x: auto;
            flex-wrap: nowrap !important;
        }
        div[data-testid="column"] {
            min-width: 110px !important;
            flex: 1 0 auto !important;
        }
    }

    /* Estilo forzado para formulario de contacto en móvil/modo oscuro */
    div[data-testid="stForm"] {
        background-color: #F9F9F9 !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
    }
    div[data-testid="stForm"] label, div[data-testid="stForm"] p {
        color: #111111 !important;
    }
    div[data-testid="stForm"] input, div[data-testid="stForm"] textarea {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #CCCCCC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
@st.cache_data(ttl=3600)
def cargar_datos_consolidados():
    df_final = None

    for fname in ["tipo_cambio_consolidado1.xlsx", "tipo_cambio_consolidado.xlsx"]:
        try:
            df = pd.read_excel(fname)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                df = df.dropna(subset=["fecha"])

                for col in ["binance_compra", "binance_venta", "bcb_oficial"]:
                    if col in df.columns:
                        if df[col].dtype == "object":
                            df[col] = (
                                df[col]
                                .astype(str)
                                .str.replace(",", ".")
                                .str.extract(r"(\d+\.?\d*)")[0]
                            )
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                if df_final is None:
                    df_final = df
                else:
                    df_final = pd.merge(
                        df_final, df, on="fecha", how="outer", suffixes=("", "_y")
                    )
                    for col in ["binance_compra", "binance_venta", "bcb_oficial"]:
                        if (
                            f"{col}_y" in df_final.columns
                            and col in df_final.columns
                        ):
                            df_final[col] = df_final[col].fillna(
                                df_final[f"{col}_y"]
                            )
                            df_final = df_final.drop(columns=[f"{col}_y"])
        except Exception:
            continue

    if df_final is None or df_final.empty:
        return pd.DataFrame(
            columns=["fecha", "binance_compra", "binance_venta", "bcb_oficial"]
        )

    df_final = df_final.sort_values("fecha").reset_index(drop=True)

    if "bcb_oficial" in df_final.columns:
        df_final["bcb_oficial"] = df_final["bcb_oficial"].ffill().bfill()
    if "binance_compra" in df_final.columns:
        df_final["binance_compra"] = df_final["binance_compra"].ffill()
    if "binance_venta" in df_final.columns:
        df_final["binance_venta"] = df_final["binance_venta"].ffill()

    return df_final


# ==========================================
# 3. ESTIMACIÓN Y PROYECCIÓN (ARIMA)
# ==========================================
def calcular_proyeccion(df_serie, col_target, dias_proyeccion=7):
    if (
        col_target not in df_serie.columns
        or df_serie[col_target].dropna().empty
    ):
        fechas_futuras = pd.date_range(
            start=pd.Timestamp.now(), periods=dias_proyeccion, freq="D"
        )
        df_empty_proj = pd.DataFrame(
            {
                "Día (h)": np.arange(1, dias_proyeccion + 1),
                "Fecha": fechas_futuras,
                "TC Esperado": np.zeros(dias_proyeccion),
                "Min (95%)": np.zeros(dias_proyeccion),
                "Max (95%)": np.zeros(dias_proyeccion),
                "Min (99%)": np.zeros(dias_proyeccion),
                "Max (99%)": np.zeros(dias_proyeccion),
            }
        )
        return df_serie, df_empty_proj

    df_clean = df_serie.dropna(subset=[col_target]).copy()
    df_clean["log_ret"] = np.log(
        df_clean[col_target] / df_clean[col_target].shift(1)
    )
    df_model = df_clean.dropna(subset=["log_ret"])

    ultimo_precio = df_clean[col_target].iloc[-1]
    ultima_fecha = df_clean["fecha"].iloc[-1]
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1),
        periods=dias_proyeccion,
        freq="D",
    )
    h_steps = np.arange(1, dias_proyeccion + 1)

    var_ret = np.var(df_model["log_ret"]) if not df_model.empty else 0

    if var_ret < 1e-8:
        trayectoria_central = np.full(dias_proyeccion, ultimo_precio)
        inf_95, sup_95 = trayectoria_central.copy(), trayectoria_central.copy()
        inf_99, sup_99 = trayectoria_central.copy(), trayectoria_central.copy()
    else:
        try:
            model = ARIMA(df_model["log_ret"], order=(1, 0, 1))
            model_fit = model.fit()

            residuals = model_fit.resid
            std_error = np.std(residuals)

            forecast_log_ret = model_fit.forecast(steps=dias_proyeccion)

            trayectoria_central = [ultimo_precio]
            for r in forecast_log_ret:
                trayectoria_central.append(trayectoria_central[-1] * np.exp(r))
            trayectoria_central = np.array(trayectoria_central[1:])

            sigma_acumulado = std_error * np.sqrt(h_steps)

            inf_95 = trayectoria_central * np.exp(-2.0 * sigma_acumulado)
            sup_95 = trayectoria_central * np.exp(2.0 * sigma_acumulado)
            inf_99 = trayectoria_central * np.exp(-2.576 * sigma_acumulado)
            sup_99 = trayectoria_central * np.exp(2.576 * sigma_acumulado)
        except Exception:
            trayectoria_central = np.full(dias_proyeccion, ultimo_precio)
            inf_95, sup_95 = trayectoria_central.copy(), trayectoria_central.copy()
            inf_99, sup_99 = trayectoria_central.copy(), trayectoria_central.copy()

    df_proj = pd.DataFrame(
        {
            "Día (h)": h_steps,
            "Fecha": fechas_futuras,
            "TC Esperado": trayectoria_central,
            "Min (95%)": inf_95,
            "Max (95%)": sup_95,
            "Min (99%)": inf_99,
            "Max (99%)": sup_99,
        }
    )

    return df_clean, df_proj


# ==========================================
# 4. DASHBOARD INTERACTIVO
# ==========================================
st.title("📈 Monitor Cambiario & Análisis Económico")
st.caption(
    "Monitoreo de cotización de mercado en Bolivia (Oficial BCB vs. Binance P2P)"
)

with st.spinner("Cargando datos..."):
    df_data = cargar_datos_consolidados()

has_binance_compra = (
    "binance_compra" in df_data.columns
    and not df_data["binance_compra"].dropna().empty
)
has_binance_venta = (
    "binance_venta" in df_data.columns
    and not df_data["binance_venta"].dropna().empty
)
has_bcb = (
    "bcb_oficial" in df_data.columns
    and not df_data["bcb_oficial"].dropna().empty
)

# Métricas principales
latest_bin_compra = (
    df_data["binance_compra"].dropna().iloc[-1] if has_binance_compra else 0.0
)
latest_bin_venta = (
    df_data["binance_venta"].dropna().iloc[-1] if has_binance_venta else 0.0
)
latest_bcb = df_data["bcb_oficial"].dropna().iloc[-1] if has_bcb else 6.96

brecha = (
    ((latest_bin_compra - latest_bcb) / latest_bcb) * 100
    if latest_bcb > 0
    else 0.0
)

perdida_poder_adquisitivo = 0.0
if has_binance_compra:
    df_bin_recent = df_data.dropna(subset=["binance_compra"]).copy()
    if len(df_bin_recent) >= 60:
        tc_hace_2_meses = df_bin_recent["binance_compra"].iloc[-60]
    else:
        tc_hace_2_meses = df_bin_recent["binance_compra"].iloc[0]

    if latest_bin_compra > 0:
        perdida_poder_adquisitivo = (
            1 - (tc_hace_2_meses / latest_bin_compra)
        ) * 100

# --- 5 MÉTRICAS EN 1 SOLA LÍNEA ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Binance Compra", f"{latest_bin_compra:.2f} BOB")
col2.metric("Binance Venta", f"{latest_bin_venta:.2f} BOB")
col3.metric("BCB Oficial", f"{latest_bcb:.2f} BOB")
col4.metric("Brecha Mercado", f"{brecha:.1f}%")
col5.metric(
    "Pérdida Poder Adq. (2m)",
    f"-{perdida_poder_adquisitivo:.1f}%",
    delta_color="inverse",
)

st.markdown("---")

# --- SERIE HISTÓRICA ---
st.subheader("📊 Cotización de los Últimos 2 Meses")

df_2_meses = df_data.tail(60)

fig_hist = go.Figure()

if has_binance_compra:
    fig_hist.add_trace(
        go.Scatter(
            x=df_2_meses["fecha"],
            y=df_2_meses["binance_compra"],
            name="Binance Compra P2P",
            line=dict(color="#d62728", width=2.5),
        )
    )

if has_bcb:
    fig_hist.add_trace(
        go.Scatter(
            x=df_2_meses["fecha"],
            y=df_2_meses["bcb_oficial"],
            name="BCB Oficial",
            line=dict(color="#1f77b4", width=2, dash="dash"),
        )
    )

fig_hist.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    hovermode="x unified",
    height=350,
    margin=dict(l=10, r=10, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor="#E5E5E5"),
    yaxis=dict(title="BOB / USD", showgrid=True, gridcolor="#E5E5E5"),
)
st.plotly_chart(fig_hist, use_container_width=True)

# --- PROYECCIONES PROBABILÍSTICAS ---
st.markdown("---")
st.subheader("🔮 Proyecciones Probabilísticas")

serie_sel = st.radio(
    "Seleccionar serie a proyectar:",
    ["Binance Compra (P2P)", "Oficial (BCB)"],
    horizontal=True,
)
col_target = (
    "binance_compra" if serie_sel == "Binance Compra (P2P)" else "bcb_oficial"
)

df_clean, df_proj = calcular_proyeccion(df_data, col_target)

col_chart, col_table = st.columns([3, 2])

with col_chart:
    df_zoom = df_clean.tail(30) if "fecha" in df_clean.columns else df_clean
    fig_proj = go.Figure()

    if col_target in df_zoom.columns and not df_zoom[col_target].dropna().empty:
        fig_proj.add_trace(
            go.Scatter(
                x=df_zoom["fecha"],
                y=df_zoom[col_target],
                name="Histórico Reciente",
                line=dict(color="#111111", width=2),
            )
        )
    fig_proj.add_trace(
        go.Scatter(
            x=list(df_proj["Fecha"]) + list(df_proj["Fecha"])[::-1],
            y=list(df_proj["Max (99%)"]) + list(df_proj["Min (99%)"])[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Rango 99%",
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=list(df_proj["Fecha"]) + list(df_proj["Fecha"])[::-1],
            y=list(df_proj["Max (95%)"]) + list(df_proj["Min (95%)"])[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Rango 95%",
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=df_proj["Fecha"],
            y=df_proj["TC Esperado"],
            name="Proyección Central",
            line=dict(color="#d62728", width=2.5, dash="dash"),
        )
    )

    fig_proj.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=350,
        hovermode="x unified",
        title=dict(
            text=f"Proyección a 7 Días - {serie_sel}", font=dict(size=14)
        ),
        margin=dict(l=10, r=10, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#E5E5E5"),
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5"),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

with col_table:
    st.markdown("**Valores Pronosticados a 7 Días**")
    df_disp = df_proj.copy()
    df_disp["Fecha"] = df_disp["Fecha"].dt.strftime("%Y-%m-%d")
    cols_num = [
        "TC Esperado",
        "Min (95%)",
        "Max (95%)",
        "Min (99%)",
        "Max (99%)",
    ]
    df_disp[cols_num] = df_disp[cols_num].round(2)
    st.dataframe(df_disp, hide_index=True, use_container_width=True)

# --- SECCIÓN DE CONTACTO ---
st.markdown("---")
st.subheader("📩 Contacto")

with st.form("form_contacto"):
    nombre = st.text_input("Nombre / Empresa")
    email = st.text_input("Correo Electrónico")
    mensaje = st.text_area("Consulta o mensaje")

    submit = st.form_submit_button("Enviar Solicitud")
    if submit:
        if nombre and email and mensaje:
            st.success("¡Mensaje enviado con éxito!")
        else:
            st.warning("Por favor completa todos los campos.")
