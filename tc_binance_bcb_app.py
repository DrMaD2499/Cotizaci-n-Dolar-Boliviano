import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Monitor Cambiario Boliviano",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Fondo general */
    .stApp {
        background-color: #FFFFFF;
        color: #111111;
    }
    header {visibility: hidden;}

    /* ===== MÉTRICAS - contraste y legibilidad ===== */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #222222 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #111111 !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* ===== Optimización móvil ===== */
    @media (max-width: 768px) {
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
        }
        div[data-testid="stMetric"] {
            padding: 10px 8px;
            margin-bottom: 6px;
        }
        div[data-testid="column"] {
            padding: 0 4px !important;
        }
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
                        if f"{col}_y" in df_final.columns and col in df_final.columns:
                            df_final[col] = df_final[col].fillna(df_final[f"{col}_y"])
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
# 3. ESTIMACIÓN Y PROYECCIÓN PROBABILÍSTICA (ARIMA)
# ==========================================
def calcular_proyeccion(df_serie, col_target, dias_proyeccion=7):
    if col_target not in df_serie.columns or df_serie[col_target].dropna().empty:
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
    df_clean["log_ret"] = np.log(df_clean[col_target] / df_clean[col_target].shift(1))
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
   
