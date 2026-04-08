# =============================================================================
# ANÁLISIS DE DIVERSIFICACIÓN INTERNACIONAL (5 MERCADOS GLOBALES)
# Proyecto Universitario de Finanzas Cuantitativas
# =============================================================================

# --- Instalación de librerías (descomenta si ejecutas en Google Colab) ---
# !pip install yfinance seaborn matplotlib numpy pandas

# =============================================================================
# BLOQUE 0: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)

print("=" * 70)
print(" ANÁLISIS DE DIVERSIFICACIÓN INTERNACIONAL - MATRICES DE CORRELACIÓN")
print("=" * 70)

# =============================================================================
# BLOQUE 1: DICCIONARIO DE ACTIVOS POR MERCADO
# 75 activos divididos en 5 mercados (15 por mercado)
# =============================================================================
MARKETS = {
    "Japón (Nikkei 225)": [
        "8035.T",
        "6758.T",
        "9984.T",
        "6857.T",
        "6702.T",
        "6503.T",
        "4689.T",
        "7203.T",
        "6301.T",
        "7011.T",
        "9101.T",
        "6501.T",
        "7267.T",
        "6902.T",
        "1801.T",
    ],
    "Alemania (DAX 40)": [
        "SAP.DE",
        "IFX.DE",
        "AFX.DE",
        "WCH.DE",
        "NEM.DE",
        "VAR1.DE",
        "SOW.DE",
        "SIE.DE",
        "DHL.DE",
        "VOW3.DE",
        "MBG.DE",
        "MTX.DE",
        "RHM.DE",
        "CON.DE",
        "HEI.DE",
    ],
    "Reino Unido (FTSE 100)": [
        "SGE.L",
        "OCDO.L",
        "RMV.L",
        "AUTO.L",
        "DARK.L",
        "SPT.L",
        "BAE.L",
        "RR.L",
        "IAG.L",
        "FGP.L",
        "MGG.L",
        "MEL.L",
        "WEIR.L",
        "RKT.L",
        "SMDS.L",
    ],
    "Hong Kong (Hang Seng)": [
        "0700.HK",
        "9988.HK",
        "3690.HK",
        "9888.HK",
        "1810.HK",
        "0285.HK",
        "1347.HK",
        "1211.HK",
        "0669.HK",
        "1928.HK",
        "0001.HK",
        "0016.HK",
        "0066.HK",
        "0386.HK",
        "1088.HK",
    ],
    "Francia (CAC 40)": [
        "CAP.PA",
        "STMPA.PA",
        "DSY.PA",
        "ATO.PA",
        "WLN.PA",
        "SESL.PA",
        "ORA.PA",
        "AIR.PA",
        "SAF.PA",
        "DG.PA",
        "SCHP.PA",
        "ALO.PA",
        "ENGI.PA",
        "SU.PA",
        "TTE.PA",
    ],
}

# =============================================================================
# BLOQUE 2: DESCARGA INDIVIDUAL Y CÁLCULO DE CORRELACIONES
# =============================================================================
retornos_por_mercado = {}
matrices_corr = {}
correlaciones_promedio = {}

print("\n[1/4] Descargando datos (10 años) y procesando retornos por mercado...")

for nombre_mercado, tickers in MARKETS.items():
    print(f"      -> Procesando {nombre_mercado}...")

    datos = yf.download(tickers, period="10y", auto_adjust=False, progress=False)
    precios = datos["Adj Close"]

    # Excluir tickers sin datos (p. ej. delisted o no disponibles en Yahoo)
    tickers_invalidos = precios.columns[precios.isna().all()].tolist()
    if tickers_invalidos:
        print(
            f"         Aviso: se excluyen {len(tickers_invalidos)} ticker(s) sin datos: "
            f"{', '.join(tickers_invalidos)}"
        )
        precios = precios.drop(columns=tickers_invalidos)
    if precios.shape[1] < 2:
        raise RuntimeError(
            f"{nombre_mercado}: datos insuficientes tras excluir tickers inválidos "
            f"({precios.shape[1]} activo(s) válido(s))."
        )

    # Relleno para alinear feriados locales y limpieza residual
    precios_limpios = precios.ffill().dropna()
    if precios_limpios.empty:
        raise RuntimeError(
            f"No se pudieron obtener precios válidos para {nombre_mercado}. "
            "Verifica la conexión a internet o la disponibilidad de Yahoo Finance."
        )

    retornos = precios_limpios.pct_change().dropna()
    if retornos.empty:
        raise RuntimeError(
            f"No fue posible calcular retornos diarios para {nombre_mercado}."
        )
    retornos_por_mercado[nombre_mercado] = retornos

    matriz = retornos.corr()
    matrices_corr[nombre_mercado] = matriz

    mascara_tri_superior = np.triu(np.ones_like(matriz, dtype=bool), k=1)
    valores_corr = matriz.where(mascara_tri_superior).values.flatten()
    valores_corr_limpios = valores_corr[~np.isnan(valores_corr)]
    if len(valores_corr_limpios) == 0:
        raise RuntimeError(
            f"No hay correlaciones cruzadas válidas para {nombre_mercado}."
        )
    correlaciones_promedio[nombre_mercado] = np.mean(valores_corr_limpios)

print("\n[2/4] Análisis de correlación promedio completado.")

# =============================================================================
# BLOQUE 3: VISUALIZACIÓN EN CUADRÍCULA (SUBPLOTS)
# =============================================================================
print("[3/4] Generando gráfico comparativo de matrices de correlación...")

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes = axes.flatten()

for i, (nombre_mercado, matriz) in enumerate(matrices_corr.items()):
    ax = axes[i]
    sns.heatmap(
        matriz,
        annot=False,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    promedio = correlaciones_promedio[nombre_mercado]
    ax.set_title(
        f"{nombre_mercado}\nCorr. Promedio: {promedio:.4f}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

fig.delaxes(axes[5])

plt.suptitle(
    "Comparativa de Matrices de Correlación por Mercado",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
plt.savefig("comparativa_mercados.png", dpi=200, bbox_inches="tight")
plt.show()
print("      Gráfico guardado como 'comparativa_mercados.png'.")

# =============================================================================
# BLOQUE 4: TABLA DE RESULTADOS Y SINCRONIZACIÓN GLOBAL
# =============================================================================
print(
    "\n[4/4] Resultados del Índice de Diversificación "
    "(menor correlación promedio = mejor):"
)
df_resultados = pd.DataFrame.from_dict(
    correlaciones_promedio, orient="index", columns=["Correlación Promedio"]
).sort_values(by="Correlación Promedio", ascending=True)
print("\n", df_resultados.to_string())

retornos_globales = pd.concat(retornos_por_mercado.values(), axis=1, join="inner")
print(
    f"\nSincronización global completada. "
    f"Días comunes para los 75 activos: {retornos_globales.shape[0]}"
)
