# =============================================================================
# FRONTERA EFICIENTE DE MARKOWITZ - SIMULACIÓN DE MONTE CARLO
# Proyecto Universitario de Finanzas Cuantitativas
# Listo para ejecutar en Google Colab
# =============================================================================

# --- Instalación de librerías (descomenta si ejecutas en Google Colab) ---
# !pip install yfinance seaborn matplotlib numpy pandas

# =============================================================================
# BLOQUE 0: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# BLOQUE 1: EXTRACCIÓN DE DATOS (Yahoo Finance)
# Descarga 10 años de precios de cierre ajustados para 15 tickers del S&P 500.
# Los activos cubren múltiples sectores: Tecnología, Industria, Salud,
# Consumo Básico, Finanzas, Energía y Servicios Públicos.
# =============================================================================

# Lista de 15 tickers diversificados del S&P 500
TICKERS = [
   'AAPL',  # Tecnología - Apple
    'MSFT',  # Tecnología - Microsoft
    'NVDA',  # Tecnología - Nvidia
    'AMZN',  # Tecnología - Amazon
    'ACN',   # Tecnología - Accenture
    'CSCO',  # Tecnología - Cisco
    'INTU',  # Tecnología - Intuit

    # Sector Industrial / Manufactura
    'CAT',   # Industria - Caterpillar
    'DE',    # Industria - Deere & Company
    'HON',   # Industria - Honeywell
    'GE',    # Industria - General Electric
    'F',     # Industria - Ford Motor
    'MMM',   # Industria - 3M Company
    'UNP',   # Industria - Union Pacific
    'UPS'    # Industria - United Parcel Service
]

print("=" * 65)
print("  FRONTERA EFICIENTE DE MARKOWITZ - SIMULACIÓN DE MONTE CARLO")
print("=" * 65)
print(f"\n[1/5] Descargando 10 años de datos históricos para {len(TICKERS)} activos...")

# Descarga de datos: 10 años, columna 'Adj Close'
datos_crudos = yf.download(TICKERS, period='10y', auto_adjust=False, progress=False)
precios = datos_crudos['Adj Close'].dropna()

print(f"      Datos descargados: {precios.shape[0]} días de trading, {precios.shape[1]} activos.")
print(f"      Período: {precios.index[0].strftime('%Y-%m-%d')} → {precios.index[-1].strftime('%Y-%m-%d')}")
print(f"      Activos disponibles: {list(precios.columns)}")

# =============================================================================
# BLOQUE 2: ANÁLISIS INICIAL Y MATRIZ DE CORRELACIÓN
# Se calculan los retornos diarios simples y se visualiza la correlación
# entre activos para justificar los beneficios de la diversificación.
# =============================================================================

print("\n[2/5] Calculando retornos diarios y generando matriz de correlación...")

# Retornos diarios simples: R_t = (P_t / P_{t-1}) - 1
retornos_diarios = precios.pct_change().dropna()

# Número de días de trading anuales (convención estándar)
DIAS_TRADING = 252

# Estadísticas de retornos
retorno_medio_diario = retornos_diarios.mean()
matriz_cov_anualizada = retornos_diarios.cov() * DIAS_TRADING

print(f"      Retornos diarios calculados: {retornos_diarios.shape[0]} observaciones.")

# --- Gráfico 1: Mapa de Calor de Correlación ---
matriz_correlacion = retornos_diarios.corr()

fig1, ax1 = plt.subplots(figsize=(12, 9))
sns.heatmap(
    matriz_correlacion,
    annot=True,
    fmt=".2f",
    cmap='RdYlGn',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor='white',
    ax=ax1,
    cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de Correlación'}
)
ax1.set_title(
    'Matriz de Correlación de Retornos Diarios\n(15 Activos Diversificados del S&P 500)',
    fontsize=14,
    fontweight='bold',
    pad=15
)
ax1.tick_params(axis='x', rotation=45, labelsize=9)
ax1.tick_params(axis='y', rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig('correlacion_activos.png', dpi=150, bbox_inches='tight')
plt.show()
print("      Matriz de correlación guardada como 'correlacion_activos.png'.")

# =============================================================================
# BLOQUE 3: SIMULACIÓN DE MONTE CARLO
# Se generan 100,000 portafolios con pesos aleatorios (suma = 1).
# Para cada uno se calculan: Rendimiento Esperado, Volatilidad y Ratio de Sharpe.
# =============================================================================

print("\n[3/5] Ejecutando Simulación de Monte Carlo (100,000 portafolios)...")

NUM_PORTAFOLIOS = 100_000
NUM_ACTIVOS = len(precios.columns)
TASA_LIBRE_RIESGO = 0.042   # Risk-Free Rate: 4.2% anual
RANDOM_SEED = 42             # Semilla para reproducibilidad
MIN_PESO = 0.001             # Umbral mínimo para mostrar pesos en reportes

# Arrays para almacenar los resultados de cada simulación
resultados = np.zeros((NUM_PORTAFOLIOS, 3))  # [rendimiento, volatilidad, sharpe]
pesos_sim = np.zeros((NUM_PORTAFOLIOS, NUM_ACTIVOS))

# Vectores de retornos medios anualizados para cálculo eficiente
retorno_medio_anual = retorno_medio_diario.values * DIAS_TRADING

np.random.seed(RANDOM_SEED)

for i in range(NUM_PORTAFOLIOS):
    # Generación de pesos aleatorios normalizados (suman exactamente 1)
    pesos_aleatorios = np.random.random(NUM_ACTIVOS)
    pesos_aleatorios /= pesos_aleatorios.sum()

    # Rendimiento esperado del portafolio (E[R_p] = w^T · μ), anualizado
    rendimiento = np.dot(pesos_aleatorios, retorno_medio_anual)

    # Volatilidad anualizada: σ_p = sqrt(w^T · Σ · w)
    varianza = pesos_aleatorios @ matriz_cov_anualizada.values @ pesos_aleatorios
    volatilidad = np.sqrt(varianza)

    # Ratio de Sharpe: (E[R_p] - R_f) / σ_p
    sharpe = (rendimiento - TASA_LIBRE_RIESGO) / volatilidad

    # Almacenamiento de resultados
    resultados[i, 0] = rendimiento
    resultados[i, 1] = volatilidad
    resultados[i, 2] = sharpe
    pesos_sim[i, :] = pesos_aleatorios

print(f"      Simulación completada. {NUM_PORTAFOLIOS:,} portafolios evaluados.")

# Conversión a DataFrame para facilitar el análisis
df_portafolios = pd.DataFrame(resultados, columns=['Rendimiento', 'Volatilidad', 'Sharpe'])

# =============================================================================
# BLOQUE 4: OPTIMIZACIÓN — PORTAFOLIOS CRÍTICOS
# Se identifican el Portafolio de Máxima Eficiencia (Max Sharpe) y el
# Portafolio de Mínima Varianza Global (GMV).
# =============================================================================

print("\n[4/5] Identificando portafolios óptimos...")

# --- Portafolio de Máximo Ratio de Sharpe ---
idx_max_sharpe = df_portafolios['Sharpe'].idxmax()
portafolio_max_sharpe = df_portafolios.loc[idx_max_sharpe]
pesos_max_sharpe = pesos_sim[idx_max_sharpe]

# --- Portafolio de Mínima Varianza Global (GMV) ---
idx_min_var = df_portafolios['Volatilidad'].idxmin()
portafolio_min_var = df_portafolios.loc[idx_min_var]
pesos_min_var = pesos_sim[idx_min_var]

# --- Reporte de resultados ---
print("\n  ┌─────────────────────────────────────────────────────────────┐")
print("  │           PORTAFOLIO DE MÁXIMO RATIO DE SHARPE              │")
print("  ├─────────────────────────────────────────────────────────────┤")
print(f"  │  Rendimiento Esperado (E[R_p]):  {portafolio_max_sharpe['Rendimiento']*100:7.2f}%                  │")
print(f"  │  Volatilidad Anualizada (σ_p):   {portafolio_max_sharpe['Volatilidad']*100:7.2f}%                  │")
print(f"  │  Ratio de Sharpe:                {portafolio_max_sharpe['Sharpe']:7.4f}                  │")
print("  ├─────────────────────────────────────────────────────────────┤")
print("  │  Pesos del portafolio:                                      │")
for ticker, peso in zip(precios.columns, pesos_max_sharpe):
    if peso > MIN_PESO:
        print(f"  │    {ticker:<6}: {peso*100:6.2f}%                                          │")
print("  └─────────────────────────────────────────────────────────────┘")

print("\n  ┌─────────────────────────────────────────────────────────────┐")
print("  │        PORTAFOLIO DE MÍNIMA VARIANZA GLOBAL (GMV)          │")
print("  ├─────────────────────────────────────────────────────────────┤")
print(f"  │  Rendimiento Esperado (E[R_p]):  {portafolio_min_var['Rendimiento']*100:7.2f}%                  │")
print(f"  │  Volatilidad Anualizada (σ_p):   {portafolio_min_var['Volatilidad']*100:7.2f}%                  │")
print(f"  │  Ratio de Sharpe:                {portafolio_min_var['Sharpe']:7.4f}                  │")
print("  ├─────────────────────────────────────────────────────────────┤")
print("  │  Pesos del portafolio:                                      │")
for ticker, peso in zip(precios.columns, pesos_min_var):
    if peso > MIN_PESO:
        print(f"  │    {ticker:<6}: {peso*100:6.2f}%                                          │")
print("  └─────────────────────────────────────────────────────────────┘")

# DataFrames con pesos detallados
df_pesos_max_sharpe = pd.DataFrame({
    'Activo': precios.columns,
    'Peso (%)': (pesos_max_sharpe * 100).round(2)
}).sort_values('Peso (%)', ascending=False).reset_index(drop=True)

df_pesos_min_var = pd.DataFrame({
    'Activo': precios.columns,
    'Peso (%)': (pesos_min_var * 100).round(2)
}).sort_values('Peso (%)', ascending=False).reset_index(drop=True)

print("\n  Tabla de pesos — Portafolio Max Sharpe:")
print(df_pesos_max_sharpe.to_string(index=False))
print("\n  Tabla de pesos — Portafolio GMV:")
print(df_pesos_min_var.to_string(index=False))

# =============================================================================
# BLOQUE 5: VISUALIZACIÓN AVANZADA — FRONTERA EFICIENTE
# Gráfico de dispersión Riesgo vs. Retorno con gradiente de color por Sharpe.
# Se marcan con estrellas los portafolios Max Sharpe y GMV.
# =============================================================================

print("\n[5/5] Generando visualización de la Frontera Eficiente...")

fig2, ax2 = plt.subplots(figsize=(14, 9))

# --- Scatter plot con gradiente de color según Ratio de Sharpe ---
scatter = ax2.scatter(
    df_portafolios['Volatilidad'] * 100,   # Eje X: Riesgo en %
    df_portafolios['Rendimiento'] * 100,   # Eje Y: Retorno en %
    c=df_portafolios['Sharpe'],            # Color según Sharpe
    cmap='viridis',
    alpha=0.4,
    s=5,
    zorder=1,
    label='_nolegend_'
)

# Barra de color (colorbar) indicando el Ratio de Sharpe
cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
cbar.set_label('Ratio de Sharpe', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

# --- Marcador: Portafolio de Máximo Ratio de Sharpe (estrella dorada) ---
ax2.scatter(
    portafolio_max_sharpe['Volatilidad'] * 100,
    portafolio_max_sharpe['Rendimiento'] * 100,
    marker='*',
    s=600,
    color='gold',
    edgecolors='black',
    linewidths=1.5,
    zorder=5,
    label=f"Max Sharpe  |  σ={portafolio_max_sharpe['Volatilidad']*100:.2f}%"
          f"  |  E[R]={portafolio_max_sharpe['Rendimiento']*100:.2f}%"
          f"  |  SR={portafolio_max_sharpe['Sharpe']:.4f}"
)

# --- Marcador: Portafolio de Mínima Varianza Global (estrella roja) ---
ax2.scatter(
    portafolio_min_var['Volatilidad'] * 100,
    portafolio_min_var['Rendimiento'] * 100,
    marker='*',
    s=600,
    color='red',
    edgecolors='black',
    linewidths=1.5,
    zorder=5,
    label=f"Min Varianza (GMV)  |  σ={portafolio_min_var['Volatilidad']*100:.2f}%"
          f"  |  E[R]={portafolio_min_var['Rendimiento']*100:.2f}%"
          f"  |  SR={portafolio_min_var['Sharpe']:.4f}"
)

# --- Anotaciones de texto sobre los portafolios críticos ---
ax2.annotate(
    'Max Sharpe',
    xy=(portafolio_max_sharpe['Volatilidad'] * 100, portafolio_max_sharpe['Rendimiento'] * 100),
    xytext=(8, 5),
    textcoords='offset points',
    fontsize=10,
    fontweight='bold',
    color='black',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.7)
)

ax2.annotate(
    'GMV',
    xy=(portafolio_min_var['Volatilidad'] * 100, portafolio_min_var['Rendimiento'] * 100),
    xytext=(8, -15),
    textcoords='offset points',
    fontsize=10,
    fontweight='bold',
    color='white',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8)
)

# --- Formato profesional del gráfico ---
ax2.set_title(
    'Frontera Eficiente de Markowitz\n'
    f'Simulación de Monte Carlo — {NUM_PORTAFOLIOS:,} Portafolios | '
    f'Risk-Free Rate: {TASA_LIBRE_RIESGO*100:.1f}% | '
    f'{len(TICKERS)} Activos S&P 500',
    fontsize=13,
    fontweight='bold',
    pad=15
)
ax2.set_xlabel('Riesgo — Volatilidad Anualizada (%)', fontsize=12, labelpad=10)
ax2.set_ylabel('Retorno Esperado Anualizado (%)', fontsize=12, labelpad=10)
ax2.legend(
    loc='upper left',
    fontsize=9,
    framealpha=0.9,
    edgecolor='gray',
    fancybox=True
)
ax2.grid(True, linestyle='--', alpha=0.5, color='gray')
ax2.set_facecolor('#f8f9fa')
fig2.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('frontera_eficiente_markowitz.png', dpi=150, bbox_inches='tight')
plt.show()
print("      Frontera Eficiente guardada como 'frontera_eficiente_markowitz.png'.")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 65)
print("  ANÁLISIS COMPLETADO")
print("=" * 65)
print(f"  • Activos analizados:          {NUM_ACTIVOS}")
print(f"  • Portafolios simulados:       {NUM_PORTAFOLIOS:,}")
print(f"  • Tasa libre de riesgo (R_f):  {TASA_LIBRE_RIESGO*100:.1f}%")
print(f"  • Días de trading anualizados: {DIAS_TRADING}")
print()
print(f"  Portafolio Max Sharpe:")
print(f"    Rendimiento:  {portafolio_max_sharpe['Rendimiento']*100:.2f}%")
print(f"    Volatilidad:  {portafolio_max_sharpe['Volatilidad']*100:.2f}%")
print(f"    Sharpe Ratio: {portafolio_max_sharpe['Sharpe']:.4f}")
print()
print(f"  Portafolio GMV (Mínima Varianza):")
print(f"    Rendimiento:  {portafolio_min_var['Rendimiento']*100:.2f}%")
print(f"    Volatilidad:  {portafolio_min_var['Volatilidad']*100:.2f}%")
print(f"    Sharpe Ratio: {portafolio_min_var['Sharpe']:.4f}")
print("=" * 65)
