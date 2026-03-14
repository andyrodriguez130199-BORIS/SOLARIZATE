# =============================================================================
# EXTRACCIÓN Y EXPORTACIÓN DE PRECIOS HISTÓRICOS A EXCEL
# 15 Activos del S&P 500 — Últimos 10 años de datos (Adj Close)
# Listo para ejecutar en Google Colab
# =============================================================================

# --- Instalación de librerías (descomenta si ejecutas en Google Colab) ---
# !pip install yfinance openpyxl pandas

# =============================================================================
# BLOQUE 1: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# BLOQUE 2: CONFIGURACIÓN — UNIVERSO DE ACTIVOS
# 15 tickers del S&P 500 seleccionados de industrias diferenciadas para
# reducir el riesgo no sistemático mediante diversificación sectorial:
#   Tecnología, Industria, Salud, Consumo Básico, Finanzas, Energía y
#   Servicios Públicos.
# =============================================================================

TICKERS = [
    'AAPL',  # Tecnología      - Apple Inc.
    'MSFT',  # Tecnología      - Microsoft Corp.
    'CAT',   # Industria       - Caterpillar Inc.
    'DE',    # Industria       - Deere & Company
    'HON',   # Industria       - Honeywell International
    'JNJ',   # Salud           - Johnson & Johnson
    'PFE',   # Salud           - Pfizer Inc.
    'PG',    # Consumo Básico  - Procter & Gamble
    'KO',    # Consumo Básico  - Coca-Cola Company
    'JPM',   # Finanzas        - JPMorgan Chase & Co.
    'V',     # Finanzas        - Visa Inc.
    'XOM',   # Energía         - ExxonMobil Corp.
    'CVX',   # Energía         - Chevron Corp.
    'NEE',   # Servicios Públicos - NextEra Energy
    'DUK',   # Servicios Públicos - Duke Energy Corp.
]

NOMBRE_ARCHIVO = 'precios_historicos_sp500.xlsx'

# =============================================================================
# BLOQUE 3: DESCARGA DE DATOS HISTÓRICOS (Yahoo Finance)
# Se descargan los últimos 10 años de precios de cierre ajustados (Adj Close).
# Adj Close incorpora los ajustes por dividendos y splits, por lo que es
# la columna correcta para calcular retornos reales.
# =============================================================================

print("=" * 60)
print("  EXPORTACIÓN DE PRECIOS HISTÓRICOS S&P 500 → EXCEL")
print("=" * 60)
print(f"\n[1/3] Descargando 10 años de datos para {len(TICKERS)} activos...")

# Descarga de precios con period='10y'; auto_adjust=False para obtener
# explícitamente la columna 'Adj Close'.
datos_crudos = yf.download(TICKERS, period='10y', auto_adjust=False, progress=False)

# Extracción exclusiva de los precios de cierre ajustados
precios_adj_close = datos_crudos['Adj Close']

# Eliminación de filas con valores nulos para mantener la tabla limpia
precios_adj_close = precios_adj_close.dropna()

print(f"      Registros descargados:  {precios_adj_close.shape[0]} días de trading")
print(f"      Activos:                {precios_adj_close.shape[1]}")
print(f"      Período:                {precios_adj_close.index[0].strftime('%Y-%m-%d')}"
      f" → {precios_adj_close.index[-1].strftime('%Y-%m-%d')}")
print(f"      Columnas (tickers):     {list(precios_adj_close.columns)}")

# Vista previa de los primeros registros
print("\n  Vista previa (primeras 3 filas):")
print(precios_adj_close.head(3).to_string())

# =============================================================================
# BLOQUE 4: EXPORTACIÓN A EXCEL
# Las fechas quedan como índice (filas) y los tickers como encabezados
# (columnas), tal como se requiere para análisis financiero posterior.
# =============================================================================

print(f"\n[2/3] Exportando tabla a '{NOMBRE_ARCHIVO}'...")

# index=True asegura que las fechas se incluyan como la primera columna del Excel
precios_adj_close.to_excel(NOMBRE_ARCHIVO, index=True)

print(f"      ✓ Archivo '{NOMBRE_ARCHIVO}' creado exitosamente.")

# =============================================================================
# BLOQUE 5: MENSAJE FINAL — INSTRUCCIONES DE DESCARGA EN GOOGLE COLAB
# =============================================================================

print(f"\n[3/3] ¡Listo!")
print("\n" + "=" * 60)
print("  CÓMO DESCARGAR EL ARCHIVO DESDE GOOGLE COLAB")
print("=" * 60)
print("""
  El archivo Excel ha sido generado en el directorio de trabajo
  de tu sesión de Colab. Para descargarlo a tu computadora:

  OPCIÓN A — Panel lateral (recomendada):
    1. Haz clic en el ícono de carpeta (📁) en la barra
       lateral izquierda de Google Colab.
    2. Busca el archivo 'precios_historicos_sp500.xlsx'
       en la lista de archivos.
    3. Haz clic derecho sobre él y selecciona
       "Descargar" del menú contextual.

  OPCIÓN B — Con código (en una nueva celda):
    from google.colab import files
    files.download('precios_historicos_sp500.xlsx')

  Nota: los archivos en Colab se eliminan al cerrar la sesión,
  así que descárgalo o guárdalo en Google Drive antes de salir.
""")
print("=" * 60)
