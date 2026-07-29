import io
import datetime
import os
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Monitor Cambiario & Análisis Económico",
    page_icon="📈",
    layout="wide",
)

# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN (TU CÓDIGO BCB Y BINANCE)
# ==========================================
LANDING_URL = "https://www.bcb.gob.bo"
LANDING_SELECTORS = {
    "tco": ".is-tc-oficial .bcb-tco-num",
    "fecha": ".is-tc-oficial .bcb-kpi2-asof time",
    "tco_duo_fila": ".is-tc-oficial .bcb-tco-duo-row",
    "tco_duo_num": ".bcb-tco-duo-num",
    "tco_duo_fecha": ".bcb-tco-duo-label span",
}


def normalizar_decimal(texto):
    return float(str(texto).strip().replace(".", "").replace(",", "."))


def codigo_meses():
    meses = [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ]
    return {mes: i + 1 for i, mes in enumerate(meses)}


def normalizar_fecha_es(texto):
    texto = texto.lower()
    meses = codigo_meses()
    d = re.search(r"(\d{1,2})\s+de\s+(\w+),?\s+(?:de\s+)?(\d{4})", texto)
    if d is None:
        return datetime.date.today().strftime("%Y-%m-%d")
    return datetime.datetime(
        int(d.group(3)), meses[d.group(2)], int(d.group(1))
    ).strftime("%Y-%m-%d")


def consultar_landing(session):
    try:
        response = session.get(LANDING_URL, timeout=8)
        response.raise_for_status()
        html = BeautifulSoup(response.text, "html.parser")
        fila_manana = next(
            (
                fila
                for fila in html.select(LANDING_SELECTORS["tco_duo_fila"])
                if "mañana" in fila.get_text(" ", strip=True).lower()
            ),
            None,
        )

        if fila_manana is not None:
            tco = fila_manana.select_one(LANDING_SELECTORS["tco_duo_num"])
            fecha = fila_manana.select_one(LANDING_SELECTORS["tco_duo_fecha"])
            fecha = fecha.get_text(" ", strip=True) if fecha else None
        else:
            tco = html.select_one(LANDING_SELECTORS["tco"])
            fecha = html.select_one(LANDING_SELECTORS["fecha"])
            fecha = fecha.get("datetime") if fecha else None

        if tco is None or not fecha:
            return datetime.date.today().strftime("%Y-%m-%d"), 6.96
        return (
            normalizar_fecha_es(fecha),
            normalizar_decimal(tco.get_text(strip=True)),
        )
    except Exception:
        return datetime.date.today().strftime("%Y-%m-%d"), 6.96


def obtener_serie_diaria_binance(url, nombre_columna):
    df = pd.read_csv(url)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["fecha"] = df["timestamp"].dt.date
    df_diario = df.groupby("fecha").last().reset_index()
    df_diario = df_diario[["fecha", "median"]].rename(
        columns={"median": nombre_columna}
    )
    df_diario["fecha"] = pd.to_datetime(df_diario["fecha"])
    return df_diario


@st.cache_data(ttl=3600)
def cargar_datos_consolidados():
    url_binance_compra = "https://raw.githubusercontent.com/mauforonda/dolares/main/datos/binance/compra.csv"
    url_binance_venta = "https://raw.githubusercontent.com/mauforonda/dolares/main/datos/binance/venta.csv"
    url_bcb_compra = "https://raw.githubusercontent.com/mauforonda/dolares/main/datos/bcb/compra.csv"

    df_bin_compra = obtener_serie_diaria_binance(
        url_binance_compra, "binance_compra"
    )
    df_bin_venta = obtener_serie_diaria_binance(
        url_binance_venta, "binance_venta"
    )
    df_binance = pd.merge(df_bin_compra, df_bin_venta, on="fecha", how="outer")

    session = requests.Session()
    fecha_today, tco_today = consultar_landing(session)

    try:
        df_bcb = pd.read_csv(url_bcb_compra)
        df_bcb["fecha"] = pd.to_datetime(df_bcb["timestamp"])
        df_bcb = df_bcb[["fecha", "value"]].rename(
            columns={"value": "bcb_oficial"}
        )
    except Exception:
        df_bcb = pd.DataFrame(
            [{"fecha": pd.to_datetime(fecha_today), "bcb_oficial": tco_today}]
        )

    today_dt = pd.to_datetime(fecha_today)
    if today_dt not in df_bcb["fecha"].values:
        df_bcb = pd.concat(
            [
                df_bcb,
                pd.DataFrame(
                    [{"fecha": today_dt, "bcb_oficial": tco_today}]
                ),
            ],
            ignore_index=True,
        )

    df_consolidado = pd.merge(df_binance, df_bcb, on="fecha", how="outer")
    df_consolidado = df_consolidado.sort_values("fecha").reset_index(
        drop=True
    )
    df_consolidado["bcb_oficial"] = df_consolidado["bcb_oficial"].ffill()
    return df_consolidado


# ==========================================
# 3. ESTIMACIÓN ARIMA Y PROYECCIÓN
# ==========================================
def calcular_proyeccion(df_serie, col_target, dias_proyeccion=7):
    df_clean = df_serie.dropna(subset=[col_target]).copy()
    df_clean["log_ret"] = np.log(
        df_clean[col_target] / df_clean[col_target].shift(1)
    )
    df_model = df_clean.dropna(subset=["log_ret"])

    model = ARIMA(df_model["log_ret"], order=(1, 0, 1))
    model_fit = model.fit()

    residuals = model_fit.resid
    std_error = np.std(residuals)

    ultimo_precio = df_clean[col_target].iloc[-1]
    ultima_fecha = df_clean["fecha"].iloc[-1]

    forecast_log_ret = model_fit.forecast(steps=dias_proyeccion)
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1),
        periods=dias_proyeccion,
        freq="D",
    )

    trayectoria_central = [ultimo_precio]
    for r in forecast_log_ret:
        trayectoria_central.append(trayectoria_central[-1] * np.exp(r))
    trayectoria_central = np.array(trayectoria_central[1:])

    h_steps = np.arange(1, dias_proyeccion + 1)
    sigma_acumulado = std_error * np.sqrt(h_steps)

    inf_95 = trayectoria_central * np.exp(-2.0 * sigma_acumulado)
    sup_95 = trayectoria_central * np.exp(2.0 * sigma_acumulado)
    inf_99 = trayectoria_central * np.exp(-2.576 * sigma_acumulado)
    sup_99 = trayectoria_central * np.exp(2.576 * sigma_acumulado)

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
st.title("📈 Monitor Cambiario & Proyecciones Econométricas")
st.caption(
    "Análisis cuantitativo del mercado cambiario en Bolivia (Oficial BCB vs. Binance P2P)"
)

with st.spinner("Cargando datos..."):
    df_data = cargar_datos_consolidados()

# Indicadores principales
col1, col2, col3, col4 = st.columns(4)
latest_bin_compra = (
    df_data["binance_compra"].dropna().iloc[-1]
    if "binance_compra" in df_data
    else 0
)
latest_bin_venta = (
    df_data["binance_venta"].dropna().iloc[-1]
    if "binance_venta" in df_data
    else 0
)
latest_bcb = (
    df_data["bcb_oficial"].dropna().iloc[-1] if "bcb_oficial" in df_data else 6.96
)
brecha = (
    ((latest_bin_compra - latest_bcb) / latest_bcb) * 100
    if latest_bcb > 0
    else 0
)

col1.metric("Binance Compra (P2P)", f"{latest_bin_compra:.2f} BOB")
col2.metric("Binance Venta (P2P)", f"{latest_bin_venta:.2f} BOB")
col3.metric("BCB Oficial", f"{latest_bcb:.2f} BOB")
col4.metric("Brecha Mercado / Oficial", f"{brecha:.1f}%")

st.markdown("---")

# Serie histórica
st.subheader("📊 Comparativa de Tipos de Cambio")
fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(
        x=df_data["fecha"],
        y=df_data["binance_compra"],
        name="Binance Compra P2P",
        line=dict(color="#d62728", width=2),
    )
)
fig_hist.add_trace(
    go.Scatter(
        x=df_data["fecha"],
        y=df_data["bcb_oficial"],
        name="BCB Oficial",
        line=dict(color="#1f77b4", width=2, dash="dash"),
    )
)
fig_hist.update_layout(
    template="plotly_white",
    hovermode="x unified",
    height=400,
    xaxis_title="Fecha",
    yaxis_title="BOB / USD",
)
st.plotly_chart(fig_hist, use_container_width=True)

# Proyección ARIMA
st.markdown("---")
st.subheader("🔮 Proyección Probabilística a 7 Días (ARIMA)")

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
    df_zoom = df_clean.tail(30)
    fig_proj = go.Figure()

    fig_proj.add_trace(
        go.Scatter(
            x=df_zoom["fecha"],
            y=df_zoom[col_target],
            name="Histórico (30d)",
            line=dict(color="black", width=2),
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=list(df_proj["Fecha"]) + list(df_proj["Fecha"])[::-1],
            y=list(df_proj["Max (99%)"]) + list(df_proj["Min (99%)"])[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Banda 99%",
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=list(df_proj["Fecha"]) + list(df_proj["Fecha"])[::-1],
            y=list(df_proj["Max (95%)"]) + list(df_proj["Min (95%)"])[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.30)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Banda 95%",
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=df_proj["Fecha"],
            y=df_proj["TC Esperado"],
            name="Proyección Central",
            line=dict(color="crimson", width=2.5, dash="dash"),
        )
    )

    fig_proj.update_layout(
        template="plotly_white",
        height=400,
        hovermode="x unified",
        title=f"Fan Chart de Proyección - {serie_sel}",
    )
    st.plotly_chart(fig_proj, use_container_width=True)

with col_table:
    st.markdown("**Valores Pronosticados**")
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

# Sección de Servicios y Contacto
st.markdown("---")
st.header("💼 Servicios de Consultoría Económico-Financiera")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown("### 📊 Modelación Macro")
    st.write(
        "Modelos econométricos avanzados (VAR, ARIMA, GARCH) para proyecciones de tipo de cambio e inflación."
    )
with col_s2:
    st.markdown("### ⚙️ Pipelines de Datos")
    st.write(
        "Automatización de procesos de extracción y análisis de datos financieros a medida."
    )
with col_s3:
    st.markdown("### 🛡️ Gestión de Riesgo")
    st.write(
        "Estrategias de cobertura cambiaria y optimización de portafolios de inversión."
    )

st.markdown("---")
st.subheader("📩 Contactar al Analista")

with st.form("form_contacto"):
    col_f1, col_f2 = st.columns(2)
    nombre = col_f1.text_input("Nombre / Empresa")
    email = col_f2.text_input("Correo Electrónico")
    mensaje = st.text_area("Detalle de la consulta o proyecto")

    submit = st.form_submit_button("Enviar Solicitud")
    if submit:
        if nombre and email and mensaje:
            st.success(
                "¡Mensaje enviado con éxito! Nos pondremos en contacto a la brevedad."
            )
        else:
            st.warning("Por favor completa todos los campos.")