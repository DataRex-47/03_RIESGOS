"""
utils.py: el ORIGINAL de las funciones reutilizables.

Es la única fuente del código. El catálogo del vault (`00_CATALOGO_FUNCIONES.md`)
solo explica qué hace cada función y cuándo usarla, y no contiene el código.
Cada proyecto recibe una copia de este archivo en `04_scripts/` y anota la versión.
Las mejoras se hacen aquí, nunca en la copia de un proyecto.

Uso:
    from utils import frecuencias_cat, graficos_eda_categoricos
    frecuencias_cat(df.select_dtypes('object'))

⚠️ CUATRO FUNCIONES NO SON IMPORTABLES TAL CUAL: dependen de variables
   globales que en su origen venían del notebook. Hay que darles parámetro:
     · limpiar_columnas_texto  → usa BASURA
     · ranking_mi              → usa x (las predictoras)
     · log                     → usa CONFIG y datetime
     · validar_schema          → usa COLUMNAS_OBLIGATORIAS
   El resto funciona al importar.
"""

from __future__ import annotations

__version__ = "1.0"
__fecha__ = "30/10/2026"

# ── Imports ─────────────────────────────────────────────────────────
import os
import re
import unicodedata
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

# ── Imports opcionales ──────────────────────────────────────────────
# Se cargan si están instalados. Si falta uno, solo fallan las funciones
# que lo usan, no el import del módulo entero.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = sns = None

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import cloudpickle
except ImportError:
    cloudpickle = None

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = HTTPAdapter = Retry = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    import sqlalchemy as sa
except ImportError:
    sa = None

try:
    from sklearn.compose import make_column_transformer
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
except ImportError:
    make_column_transformer = confusion_matrix = make_pipeline = FunctionTransformer = None




# ────────────────────────────────────────────────────────────────────
# 1  Diagnóstico y Calidad de Datos
# ────────────────────────────────────────────────────────────────────


def reporte_salud(df: pd.DataFrame, nombre: str = "DataFrame") -> None:
    """Diagnóstico completo antes de procesar. Copiar en cualquier notebook."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  DIAGNÓSTICO: {nombre}")
    print(sep)
    print(f"  Filas:      {df.shape[0]:,}")
    print(f"  Columnas:   {df.shape[1]}")
    print(f"  RAM:        {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  Duplicados: {df.duplicated().sum():,}")

    print(f"\n  TIPOS Y NULOS:")
    for col in df.columns:
        nulos = df[col].isna().sum()
        pct   = 100 * nulos / len(df)
        flag  = " ⚠️" if nulos > 0 else ""
        print(f"    {col:<25} {str(df[col].dtype):<15} nulos: {nulos:,} ({pct:.1f}%){flag}")

    print(f"\n  ESTADÍSTICOS NUMÉRICOS:")
    for col in df.select_dtypes("number").columns:
        s = df[col]
        print(f"    {col:<20} mean={s.mean():.2f}  median={s.median():.2f}  "
              f"std={s.std():.2f}  skew={s.skew():.2f}  "
              f"min={s.min():.2f}  max={s.max():.2f}  "
              f"p99={s.quantile(.99):.2f}")

    print(f"\n  COLUMNAS CATEGÓRICAS (top 3 valores):")
    for col in df.select_dtypes("object").columns:
        top = df[col].value_counts().head(3).to_dict()
        print(f"    {col}: {top}")
    print(f"{sep}\n")


def comparar_memoria(df: pd.DataFrame, cols: list) -> None:
    antes  = df[cols].memory_usage(deep=True).sum() / 1e6
    df_cat = df[cols].astype("category")
    despues = df_cat.memory_usage(deep=True).sum() / 1e6
    print(f"Object:    {antes:.2f} MB")
    print(f"Category:  {despues:.2f} MB")
    print(f"Ahorro:    {(1 - despues/antes):.0%}")


def validar_schema(df, columnas_requeridas):
    columnas_faltantes = []
    for columna in columnas_requeridas:
        if columna not in df.columns:
            columnas_faltantes.append(columna)
    if columnas_faltantes:
        raise ValueError(f"Schema inválido. Columnas faltantes: {columnas_faltantes}")
    return True


def estandarizar_columnas(df):
    cols = []
    for col in df.columns:
        col = col.lower()
        col = unicodedata.normalize('NFKD', col).encode('ASCII','ignore').decode()
        col = re.sub(r'[–\s]', '_', col)
        col = re.sub(r'[^\w%]', '_', col)
        col = re.sub(r'__+', '_', col).strip('_')
        cols.append(col)
    df.columns = cols
    return df


def limpiar_columnas_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza todas las columnas de texto:
    1. Strip + Upper
    2. Reemplaza valores basura por NaN
    Equivalente al Patrón Polars-01 Limpieza Masiva.
    """
    df = df.copy()
    cols_texto = df.select_dtypes("object").columns

    # Paso 1: strip + upper, columna a columna
    for col in cols_texto:
        df[col] = df[col].str.strip().str.upper()

    # Paso 2: reemplazar basura por NaN
    df[cols_texto] = df[cols_texto].replace(BASURA, np.nan)

    return df



# ────────────────────────────────────────────────────────────────────
# 2  Limpieza, Nulos y Atípicos
# ────────────────────────────────────────────────────────────────────


def marcar_nulos(df):
    # Vector lógico con las variables que tienen algún nulo
    con_nulos = df.isna().sum() > 0

    # DataFrame solo con esas variables
    # .copy() para trabajar sobre un objeto propio y no sobre una vista del
    # original (evita el SettingWithCopyWarning al reescribir las columnas)
    df_con_nulos = df.loc[:, con_nulos].copy()

    # 1 si era nulo, 0 si no — columna a columna, con bucle tradicional
    for variable in df_con_nulos.columns:
        df_con_nulos[variable] = np.where(df_con_nulos[variable].isna(), 1, 0)

    # Sufijo para distinguirlas
    df_con_nulos = df_con_nulos.add_suffix('_nulos')

    # Devuelve el original con las nuevas columnas pegadas
    return pd.concat([df, df_con_nulos], axis='columns')


def rellenar_con_moda_safe(col):
    moda = col.mode()
    return col.fillna(moda[0]) if not moda.empty else col


def imputacion_relacional(df, col_objetivo, col_referencia, valor_default="Unknown"):
    """
    Rellena valores nulos en col_objetivo usando col_referencia como guía.
    Si no hay referencia disponible, usa valor_default.
    """
    mapeo = df.dropna(subset=[col_objetivo]).set_index(col_referencia)[col_objetivo].to_dict()

    # .map() traduce la referencia a su valor; el primer fillna solo toca los nulos
    # de la objetivo, y el segundo cierra los que la referencia no supo resolver.
    traduccion = df[col_referencia].map(mapeo)
    df[col_objetivo] = df[col_objetivo].fillna(traduccion).fillna(valor_default)
    return df


def agrupar_cat_raras(variable, criterio=0.05):
    # Frecuencias relativas de cada categoría
    frecuencias = variable.value_counts(normalize=True)

    # Categorías por debajo del criterio (con bucle tradicional)
    categorias_raras = frecuencias.loc[frecuencias < criterio].index.values

    temp = []
    for cada in categorias_raras:
        temp.append(cada)

    # Recodificar esas en 'OTROS', dejar el resto igual
    temp2 = np.where(variable.isin(temp), 'OTROS', variable)

    return pd.Series(temp2)


def atipicos_desv_tip(variable, x=4):
    variable = variable.dropna()          # los nulos, fuera por ahora

    media = np.mean(variable)
    sd = np.std(variable)
    umbral = sd * x

    lim_inf = media - umbral
    lim_sup = media + umbral

    # Índices de los que caen fuera de los límites (con bucle tradicional)
    indices = []

    # .items() recorre la Serie devolviendo pares (índice, valor)
    for indice, valor in variable.items():
        if valor < lim_inf or valor > lim_sup:
            indices.append(indice)

    return indices


def atipicos_cuantiles(variable, q_inf=0.25, q_sup=0.75, k=1.5):
    """
    Devuelve los índices de los valores fuera de los límites calculados con cuantiles.

    Por defecto es la regla de Tukey (la del boxplot): se pasa de 1,5 veces el
    rango intercuartílico (IQR = Q3 - Q1) por debajo de Q1 o por encima de Q3.
    Para recortar por percentiles fijos (por ejemplo el 1 % y el 99 %), usar
    q_inf=0.01, q_sup=0.99 y k=0.
    """
    variable = variable.dropna()          # los nulos, fuera por ahora

    # ── Límites a partir de los cuantiles ─────────────────────────────
    cuantil_inf = variable.quantile(q_inf)
    cuantil_sup = variable.quantile(q_sup)
    rango = cuantil_sup - cuantil_inf

    lim_inf = cuantil_inf - k * rango
    lim_sup = cuantil_sup + k * rango

    # ── Índices de los que caen fuera (con bucle tradicional) ─────────
    indices = []

    for indice, valor in variable.items():
        if valor < lim_inf or valor > lim_sup:
            indices.append(indice)

    return indices


def contar_atipicos(df, variable, num_desv_tip=4, metodo='desv_tip', k=1.5):
    """
    Cuenta cuántas veces aparece cada valor atípico de una variable.
    metodo='desv_tip' usa atipicos_desv_tip (num_desv_tip desviaciones típicas).
    metodo='cuantiles' usa atipicos_cuantiles (k veces el IQR).
    """
    if metodo == 'desv_tip':
        indices = atipicos_desv_tip(df[variable], num_desv_tip)
    elif metodo == 'cuantiles':
        indices = atipicos_cuantiles(df[variable], k=k)
    else:
        raise ValueError("metodo debe ser 'desv_tip' o 'cuantiles'")

    return df.loc[indices, variable].value_counts().sort_index()


def convertir_a_int(df, columnas):
    """
    Convierte las columnas indicadas a entero.
    Falla si aún quedan nulos: hay que imputar antes.
    """
    df = df.copy()

    for columna in columnas:
        df[columna] = df[columna].astype(int)

    return df


def eliminar_indices(indices, *dataframes):
    """
    Elimina los mismos registros (por índice) de varios DataFrames a la vez.
    Sirve cuando df, cat y num deben seguir teniendo las mismas filas.
    Devuelve una tupla con los DataFrames en el mismo orden en que entraron.
    """
    resultado = []

    for df in dataframes:
        resultado.append(df[~df.index.isin(indices)])

    return tuple(resultado)



# ────────────────────────────────────────────────────────────────────
# 3  EDA Automatizado
# ────────────────────────────────────────────────────────────────────


def frecuencias_cat(df_cat):
    largo = df_cat.astype(str).melt(var_name='variable', value_name='Valor')
    resultado = largo.groupby('variable')['Valor'].value_counts(normalize=True).rename('Frecuencia').reset_index()
    return resultado.sort_values(['variable', 'Frecuencia'], ascending=False).reset_index(drop=True)


def estadisticos_cont(df_cont):
    # describe() traspuesto: una fila por variable
    estadisticos = df_cont.describe().T

    # Añadir la mediana
    estadisticos['median'] = df_cont.median()

    # Reordenar para que la mediana quede pegada a la media
    estadisticos = estadisticos.iloc[:, [0, 1, 8, 2, 3, 4, 5, 6, 7]]

    return estadisticos


def graficos_eda_categoricos(df_cat):

    # Filas necesarias para repartir las variables en 2 columnas
    filas = ceil(df_cat.shape[1] / 2)

    # Rejilla de subgráficos — el alto crece con el número de filas
    f, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas * 6))

    # Aplanar para recorrer la rejilla como si fuera de 1 dimensión
    ax = ax.flat

    # Un gráfico de barras horizontales por variable
    for cada, variable in enumerate(df_cat):
        df_cat[variable].value_counts().plot.barh(ax=ax[cada])
        ax[cada].set_title(variable, fontsize=12, fontweight='bold')
        ax[cada].tick_params(labelsize=12)


def analisis_distribucion_continua(df, variable, bins=150, lin_inf=None, lin_sup=None):
    """Diagnóstico de una variable continua: gráficos + tabla resumen.

    Recomendada para variables con cola larga (ingresos, importes) donde un
    histograma lineal no se ve bien. Dibuja, en una sola figura:

      - histograma lineal con mediana y media marcadas;
      - histograma con frecuencia en escala logarítmica: revela el
        comportamiento en la zona de valores bajos, que el lineal aplasta;
      - histograma de ``log10`` con la curva normal teórica superpuesta:
        sirve para decidir si una transformación logarítmica es adecuada;
      - ECDF en escala log: cuantiles exactos sin depender del ancho de bin.

    Si se pasan ``lin_inf`` y ``lin_sup``, añade a cada eje la zona fuera de
    rango sombreada y devuelve en la tabla el recuento y el porcentaje de
    valores afectados, útil para decidir recortes por percentiles.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la variable.
    variable : str
        Nombre de la columna continua a analizar.
    bins : int
        Número de bins de los histogramas (por defecto 150).
    lin_inf, lin_sup : float, opcional
        Límites inferior y superior del rango válido de negocio.

    Returns
    -------
    pandas.DataFrame
        Tabla de estadísticos: n, nulos, ceros, cero_pct, negativos, min,
        q25, mediana, media, q75, q99, max, asimetria, curtosis, media_log
        y desv_log.
    """
    from scipy import stats

    serie = df[variable].dropna()
    valores = serie.to_numpy()

    # ── Estadísticos ────────────────────────────────────────────────────
    n_total = int(df[variable].shape[0])
    n = int(serie.shape[0])
    n_nulos = n_total - n
    n_ceros = int((serie == 0).sum())
    n_negativos = int((serie < 0).sum())

    q25, mediana, q75 = np.percentile(valores, [25, 50, 75])
    q01, q99 = np.percentile(valores, [1, 99])
    media = float(np.mean(valores))
    asimetria = float(stats.skew(valores))
    curtosis = float(stats.kurtosis(valores))

    positivos = valores[valores > 0]
    if positivos.shape[0] > 2:
        media_log = float(np.mean(np.log(positivos)))
        desv_log = float(np.std(np.log(positivos)))
    else:
        media_log = np.nan
        desv_log = np.nan

    tabla = pd.DataFrame(
        {
            "valor": [
                n_total, n, n_nulos, n_ceros, n_negativos,
                float(np.min(valores)), float(q01), float(q25), float(mediana),
                media, float(q75), float(q99), float(np.max(valores)),
                asimetria, curtosis, media_log, desv_log,
            ]
        },
        index=[
            "n_total", "n_validos", "nulos", "ceros", "negativos",
            "min", "q01", "q25", "mediana", "media", "q75", "q99", "max",
            "asimetria", "curtosis", "log_media", "log_desv",
        ],
    )

    if lin_inf is not None and lin_sup is not None:
        fuera_rango = int(((serie < lin_inf) | (serie > lin_sup)).sum())
        tabla.loc["fuera_rango_n"] = fuera_rango
        tabla.loc["fuera_rango_pct"] = 100.0 * fuera_rango / n if n else np.nan

    # ── Escalas auxiliares ──────────────────────────────────────────────
    positivos_grafico = valores[valores > 0]
    hay_log = positivos_grafico.shape[0] > 1
    x_inf = lin_inf if lin_inf is not None else float(q01)
    x_sup = lin_sup if lin_sup is not None else float(q99)

    # Rangos con zoom antes y después de un rango de recorte
    rango_zoom = float(np.percentile(valores, 40))
    if rango_zoom <= 0:
        rango_zoom = float(np.percentile(positivos_grafico, 40)) if hay_log else float(q25)
    rango_zoom = max(rango_zoom, float(np.min(positivos_grafico)) if hay_log else float(q25))
    x_zoom_inf = rango_zoom * 0.85 if lin_inf is None else lin_inf * 0.85
    x_zoom_sup = rango_zoom * 1.15 if lin_sup is None else lin_sup * 1.15

    # ── Figura ──────────────────────────────────────────────────────────
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 11))
    f.suptitle(f"Distribución de {variable}", fontsize=14, fontweight="bold")

    # 1) Histograma lineal
    eje = ax[0, 0]
    sns.histplot(valores, bins=bins, ax=eje, color="steelblue")
    eje.axvline(mediana, color="crimson", linestyle="--", linewidth=1.6, label=f"mediana = {mediana:,.0f}")
    eje.axvline(media, color="darkorange", linestyle="-", linewidth=1.6, label=f"media = {media:,.0f}")
    if lin_inf is not None and lin_sup is not None:
        eje.axvspan(float(np.min(valores)), lin_inf, color="red", alpha=0.12, label=f"< {lin_inf:,.0f}")
        eje.axvspan(lin_sup, float(np.max(valores)), color="red", alpha=0.12, label=f"> {lin_sup:,.0f}")
    eje.set_title(f"Lineal — media vs mediana (desviación {media - mediana:,.0f})", fontsize=11)
    eje.set_xlabel(variable)
    eje.set_ylabel("Frecuencia")
    eje.legend(fontsize=9)

    # 2) Histograma con zoom, para ver el inicio de la distribución
    eje = ax[0, 1]
    sns.histplot(valores, bins=bins, ax=eje, color="steelblue")
    eje.set_xlim(x_zoom_inf, x_zoom_sup)
    if lin_inf is not None and lin_sup is not None:
        eje.axvspan(min(float(np.min(valores)), x_zoom_inf), lin_inf, color="red", alpha=0.12)
        eje.axvspan(lin_sup, max(float(np.max(valores)), x_zoom_sup), color="red", alpha=0.12)
    eje.set_title("Zoom en los valores bajos (recorte de cola)", fontsize=11)
    eje.set_xlabel(variable)
    eje.set_ylabel("Frecuencia")

    # 3) Histograma log10 con normal teórica superpuesta
    eje = ax[1, 0]
    if hay_log:
        log_valores = np.log10(positivos_grafico)
        sns.histplot(log_valores, bins=bins, ax=eje, color="teal", stat="density")
        x_normal = np.linspace(float(np.min(log_valores)), float(np.max(log_valores)), 300)
        eje.plot(
            x_normal,
            stats.norm.pdf(x_normal, loc=float(np.mean(log_valores)), scale=float(np.std(log_valores))),
            color="black",
            linewidth=2,
            label="normal teórica",
        )
        marcador = float(np.mean(log_valores)) + 2 * float(np.std(log_valores))
        eje.axvline(marcador, color="red", linestyle="--", linewidth=1.6, label="log10(media+2σ)")
        eje.legend(fontsize=9)
        eje.set_title("log10(valor) frente a normal teórica", fontsize=11)
        eje.set_xlabel(f"log10({variable})")
        eje.set_ylabel("Densidad")
    else:
        eje.axis("off")

    # 4) ECDF en escala log
    eje = ax[1, 1]
    if hay_log:
        ordenados = np.sort(positivos_grafico)
        acumulada = np.arange(1, ordenados.shape[0] + 1) / ordenados.shape[0]
        eje.plot(ordenados, acumulada, color="steelblue", linewidth=1.6)
        for cuantil in [0.5, 0.9, 0.99, 0.999]:
            valor_cuantil = float(np.quantile(ordenados, cuantil))
            eje.axvline(valor_cuantil, color="gray", linestyle=":", linewidth=1.2)
            eje.text(valor_cuantil, 0.02, f" P{cuantil * 100:g}", rotation=90, fontsize=8, va="bottom")
        eje.set_xscale("log")
        eje.set_title("ECDF (escala log) — cuantiles exactos", fontsize=11)
        eje.set_xlabel(variable)
        eje.set_ylabel("Proporción acumulada")
        eje.grid(alpha=0.3)
    else:
        eje.axis("off")

    plt.tight_layout()
    plt.show()

    return tabla


def graficos_eda_continuas(df_cont):

    # Filas necesarias para repartir las variables en 2 columnas
    filas = ceil(df_cont.shape[1] / 2)

    # Rejilla de subgráficos
    f, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas * 6))

    # Aplanar para recorrer la rejilla como si fuera de 1 dimensión
    ax = ax.flat

    # Un gráfico de densidad por variable
    for cada, variable in enumerate(df_cont):
        df_cont[variable].plot.density(ax=ax[cada])
        ax[cada].set_title(variable, fontsize=12, fontweight='bold')
        ax[cada].tick_params(labelsize=12)



# ────────────────────────────────────────────────────────────────────
# 4  Fechas y Series Temporales
# ────────────────────────────────────────────────────────────────────


def diferencia_temporal(fecha1, fecha2):
    # Ambas deben ser datetime para poder restarse
    fecha1 = pd.to_datetime(fecha1)
    fecha2 = pd.to_datetime(fecha2)

    diferencia = fecha2 - fecha1
    return diferencia


def componentes_fecha(df):
    componentes = pd.DataFrame()
    componentes['mes'] = df.index.month   # captura estacionalidad anual
    componentes['dia'] = df.index.day     # captura patrones intra-mes
    return(componentes)


def crear_lags(df, num_lags=7):

    lags = pd.DataFrame()

    # ── Un lag por cada período hacia atrás ────────────────────────────
    # shift(1) → valor de ayer · shift(2) → de anteayer · etc.
    # Con 7 lags capturamos toda la semana anterior, útil cuando hay
    # patrón semanal (los lunes se parecen a otros lunes)
    for cada in range(1, num_lags+1):
        # .iloc[:,0] extrae la Serie 1D del DataFrame de una columna,
        # para que encaje limpiamente en la columna nueva (ver nota abajo)
        lags['lag_'+str(cada)] = df.shift(cada).iloc[:, 0]

    return(lags)


def media_movil(df, num_periodos=7):

    mm = pd.DataFrame()

    # ── Empieza en 2, NO en 1 — ver la nota sobre redundancia abajo ────
    for cada in range(2, num_periodos+1):
        # Cadena de decisiones encadenadas, se lee de izquierda a derecha:
        #   1. .shift(1)    → nos situamos en el pasado (ver WARNING abajo)
        #   2. .rolling(cada) → abre la ventana de los 'cada' períodos previos
        #   3. .mean()      → promedia los valores dentro de esa ventana
        #   4. .iloc[:,0]   → extrae la Serie 1D, igual que en crear_lags()
        mm['mm_' + str(cada)] = df.shift(1).rolling(cada).mean().iloc[:, 0]

    return(mm)


def media_movil_ponderada(datos):
    return (datos.iloc[0]*0.03 + datos.iloc[1]*0.07 +
            datos.iloc[2]*0.20 + datos.iloc[3]*0.30 + datos.iloc[4]*0.40)


def regularizar_grupo(g):
    # Rejilla diaria completa: asfreq abre una fila por cada dia que faltaba
    g_diario = g.set_index('fecha').sort_index().asfreq('D')

    # El producto es constante dentro del grupo, se propaga tal cual
    g_diario['producto'] = g['producto'].iloc[0]

    # Los huecos que abrio asfreq se interpolan; bfill y ffill cierran los extremos
    demanda = g_diario['demanda_unidades'].interpolate()
    demanda = demanda.bfill()
    demanda = demanda.ffill()
    g_diario['demanda_unidades'] = demanda

    return g_diario


# ══════════════════════════════════════════════════════════════════════
# FUNCIÓN: rotura_stock
# ══════════════════════════════════════════════════════════════════════
# QUÉ HACE, EN CRISTIANO:
#   Mira los últimos "n" días de ventas de UNA serie (un solo producto en
#   una sola tienda). Si TODOS esos n días seguidos fueron cero ventas,
#   apunta un 1 (sospecha de rotura de stock). Si algún día hubo venta,
#   apunta un 0.
#
# CÓMO SE USA:
#   No se llama directamente sobre todo el dataframe, se llama DENTRO de
#   groupby().transform(). Por eso "ventas" aquí NO es toda la columna del
#   dataset: es la columna de ventas de UN ÚNICO par tienda-producto.
#   Y lo que devuelve es otra columna del MISMO tamaño (un 0/1 por día).

def rotura_stock(ventas, n):
    # ── PASO 1: marcar los días que vendieron CERO ────────────────────
    # (ventas == 0) devuelve True donde no hubo venta y False donde sí.
    # .astype(int) convierte ese True/False en 1/0.
    # Resultado: 1 = día sin ninguna venta, 0 = día con venta.
    es_cero = (ventas == 0).astype(int)

    # ── PASO 2: contar cuántos ceros hay en los últimos n días ────────
    # .rolling(n) crea una "ventana móvil" de n días (el día actual más
    # los n-1 anteriores). .sum() suma los 1s que caen dentro de esa
    # ventana. El resultado es un número entre 0 y n para cada día.
    ceros_en_ventana = es_cero.rolling(n).sum()

    # ── PASO 3: decidir si es rotura ──────────────────────────────────
    # Si la suma es exactamente n, es que los n días seguidos estuvieron
    # a cero => rotura confirmada => 1. En cualquier otro caso => 0.
    return (ceros_en_ventana == n).astype(int)


def estadistico_movil_serie(ventas, n, estadistico):
    # Mínimo, media o máximo de los últimos n días de UNA serie.
    # shift(1) para mirar solo al pasado: el día de hoy nunca entra en la ventana.
    return ventas.shift(1).rolling(n).agg(estadistico)


def media_movil_ponderada_serie(ventas, n):
    # Media móvil ponderada exponencial (ewm) de UNA serie: lo reciente pesa más.
    # min_periods=n deja vacíos los primeros días, igual que rolling.
    return ventas.shift(1).ewm(span=n, min_periods=n).mean()



# ────────────────────────────────────────────────────────────────────
# 5  Texto
# ────────────────────────────────────────────────────────────────────


def quitar_tildes(palabra):
    # Versión con tildes y su equivalente sin ellas, carácter a carácter
    con = 'áéíóúüñÁÉÍÓÚÜÑ'
    sin = 'aeiouunAEIOUUN'

    traductor = str.maketrans(con, sin)

    return palabra.translate(traductor)



# ────────────────────────────────────────────────────────────────────
# 6  Preselección de Variables
# ────────────────────────────────────────────────────────────────────


def ranking_mi(mutual_selector, modo='tabla'):
    # Monta el DataFrame emparejando cada valor con su nombre de variable
    ranking_mi = pd.DataFrame(mutual_selector, index=x.columns).reset_index()
    ranking_mi.columns = ['variable', 'importancia_mi']

    # Ordena de mayor a menor y añade la posición en el ranking
    ranking_mi = ranking_mi.sort_values(by='importancia_mi', ascending=False)
    ranking_mi['ranking_mi'] = np.arange(0, ranking_mi.shape[0])

    if modo == 'tabla':
        return ranking_mi
    else:
        # Gráfico de barras con el top 15
        g = ranking_mi[0:15].importancia_mi.sort_values().plot.barh()
        g.set_yticklabels(ranking_mi[0:15].sort_values(by='importancia_mi').variable)
        return g


def ranking_per(predictoras, permutacion):
    # importances_mean = cuánto empeora de media la métrica (scoring) al desordenar cada variable,
    # promediando las n_repeats veces que se desordenó (AUC en clasificación, MAE en regresión)
    ranking_per = pd.DataFrame({'variable': predictoras.columns,
                                'importancia_per': permutacion.importances_mean}) \
                    .sort_values(by='importancia_per', ascending=False)

    ranking_per['ranking_per'] = np.arange(0, ranking_per.shape[0])
    return ranking_per


def ranking_tot(rank_mi, rank_rfe, rank_per):
    # Une los tres rankings por la columna 'variable'
    ranking_tot = pd.merge(pd.merge(rank_mi, rank_rfe), rank_per)

    # Suma de posiciones: menos puntos = mejor situada en los tres
    ranking_tot['puntos'] = (ranking_tot.ranking_mi
                             + ranking_tot.ranking_rfe
                             + ranking_tot.ranking_per)

    ranking_tot.sort_values(by='puntos', inplace=True)
    ranking_tot['ranking_tot'] = range(0, len(ranking_tot.variable))
    return ranking_tot


def correlaciones_fuertes(df, lim_inf=0.3, lim_sup=1, drop_dupli=True):
    # Calcula la matriz de correlación
    c = df.corr().abs()
    # Lo pasa todo a filas con melt (equivalente a unstack)
    c = c.reset_index().melt(id_vars='index', var_name='var2', value_name='corr')
    c.columns = ['var1', 'var2', 'corr']
    # Filtra límites y ordena en descendente
    c = c.loc[(c['corr'] > lim_inf) & (c['corr'] < lim_sup), :].sort_values(by='corr', ascending=False)
    # Desduplica si se pide
    c = c if drop_dupli == False else c.drop_duplicates(subset=['corr'])
    return c



# ────────────────────────────────────────────────────────────────────
# 7  Evaluación de Modelos
# ────────────────────────────────────────────────────────────────────


def plot_roc_multi_np(y, probas, labels=None, show_auc=True):
    y = np.asarray(y).ravel()
    P = y.sum()
    N = len(y) - P

    if not isinstance(probas, (list, tuple)):
        probas = [probas]
    if labels is None:
        labels = []
        for i in range(len(probas)):
            labels.append(f"Modelo {i+1}")

    plt.figure()
    for p, lab in zip(probas, labels):
        p = np.asarray(p)
        if p.ndim == 2:
            p = p[:, 1]
        order = np.argsort(-p)
        y_sorted = y[order]

        tp = np.cumsum(y_sorted)
        fp = np.arange(1, len(y_sorted)+1) - tp

        tpr = np.r_[0, tp / max(P, 1)]
        fpr = np.r_[0, fp / max(N, 1)]

        if show_auc:
            auc_val = np.trapezoid(tpr, fpr)   # área bajo la curva por trapecios
            plt.plot(fpr, tpr, label=f"{lab} (AUC={auc_val:.3f})")
        else:
            plt.plot(fpr, tpr, label=lab)

    plt.plot([0, 1], [0, 1], "--", label="Aleatorio")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend()
    plt.show();


def plot_gain_multi(y, probas, labels=None):
    y = np.asarray(y).ravel()
    if labels is None:
        labels = []
        for i in range(len(probas)):
            labels.append(f"Modelo {i+1}")

    n = len(y)
    x_pct = np.arange(1, n+1) / n

    plt.figure()
    for proba, lab in zip(probas, labels):
        p = np.asarray(proba)
        if p.ndim == 2:
            p = p[:, 1]
        order = np.argsort(-p)
        y_sorted = y[order]
        gain = np.cumsum(y_sorted) / max(y.sum(), 1)
        plt.plot(x_pct, gain, label=lab)

    plt.plot([0, 1], [0, 1], "--", label="Aleatoria")
    plt.xlabel("% observaciones"); plt.ylabel("% objetivo acumulado")
    plt.title("Curva de Ganancia Acumulada")
    plt.legend()
    plt.grid(True)
    plt.show();


def plot_lift_multi(y, probas, labels=None):
    y = np.asarray(y).ravel()
    if labels is None:
        labels = []
        for i in range(len(probas)):
            labels.append(f"Modelo {i+1}")

    n = len(y)
    x_pct = np.arange(1, n+1) / n

    plt.figure()
    for proba, lab in zip(probas, labels):
        p = np.asarray(proba)
        if p.ndim == 2:
            p = p[:, 1]
        order = np.argsort(-p)
        y_sorted = y[order]
        gain = np.cumsum(y_sorted) / max(y.sum(), 1)
        lift = gain / x_pct
        plt.plot(x_pct, lift, label=lab)

    plt.axhline(1, linestyle="--", color="gray", label="Aleatorio")
    plt.xlabel("% observaciones"); plt.ylabel("Lift")
    plt.title("Curva de Lift")
    plt.legend()
    plt.grid(True)
    plt.show();


def reporting_regresion(real, pred, n_tramos=5, residuos=False):
    """Reporting de un regresor, con los gráficos uno debajo de otro y una tabla por tramos
    del valor real. Sirve igual con porcentajes que con importes.
    residuos=True añade el histograma de residuos."""
    real = np.asarray(real).ravel()
    pred = np.asarray(pred).ravel()
    residuo = real - pred

    paneles = ['nube', 'densidad'] + (['residuos'] if residuos else [])
    fig, ejes = plt.subplots(len(paneles), 1, figsize=(9, 5 * len(paneles)))
    ejes = np.atleast_1d(ejes)

    for eje, panel in zip(ejes, paneles):
        if panel == 'nube':
            # Real contra predicho: si el modelo acierta, la nube se pega a la diagonal roja.
            # Una franja horizontal significa que predice casi lo mismo para todos.
            eje.scatter(real, pred, alpha=0.3, s=8)
            lim = [min(real.min(), pred.min()), max(real.max(), pred.max())]
            eje.plot(lim, lim, 'r--', label='Predicción perfecta')
            eje.set_xlabel('Valor real'); eje.set_ylabel('Valor predicho')
            eje.set_title('Real contra predicho'); eje.legend()
        elif panel == 'densidad':
            # Si la curva predicha es mucho más estrecha que la real,
            # el modelo se agrupa en torno a la media y no llega a los extremos.
            sns.kdeplot(real, ax=eje, label='Real')
            sns.kdeplot(pred, ax=eje, label='Predicho')
            eje.set_title('Distribución real y predicha'); eje.legend()
        else:
            # Residuo = real menos predicho. Positivo: el modelo se queda corto. Negativo: se pasa.
            eje.hist(residuo, bins=30, edgecolor='black')
            eje.axvline(0, color='r', linestyle='--')
            eje.set_xlabel('Residuo (real menos predicho)'); eje.set_title('Residuos')
        eje.grid(True)

    plt.tight_layout()
    plt.show();

    # Tabla: el error medio en cada tramo (misma cantidad de casos por tramo)
    tramos = pd.qcut(real, n_tramos, duplicates='drop')
    tabla = (pd.DataFrame({'real': real, 'pred': pred, 'error_abs': np.abs(residuo)})
               .groupby(tramos, observed=True)
               .agg(casos=('real', 'size'), real_medio=('real', 'mean'),
                    pred_medio=('pred', 'mean'), error_medio=('error_abs', 'mean'))
               .round(3))
    mae = np.abs(residuo).mean()
    print(f"MAE: {mae:.3f}  |  error medio sobre el real medio: {mae / abs(real.mean()):.1%}")
    return tabla


def de_logit_a_prob(modelo, test_x):
    logit = modelo.intercept_ + np.sum(modelo.coef_ * test_x, axis=1)
    return 1 / (1 + np.exp(-logit))



# ────────────────────────────────────────────────────────────────────
# 8  Explotación de Negocio
# ────────────────────────────────────────────────────────────────────


def max_roi(real, scoring, salida='grafico'):

    #DEFINIMOS LA MATRIZ DE IMPACTO
    ITN = 0      # Impacto de los True Negatives: ni coste ni ingreso
    IFP = -15    # Impacto de los False Positives: coste de la llamada
    IFN = -85    # Impacto de los False Negatives: coste de oportunidad de la venta perdida
    ITP = 85     # Impacto de los True Positives: 100 de ingreso - 15 de coste

    #DEFINIMOS LA FUNCION DEL VALOR ESPERADO
    def valor_esperado(matriz_conf):
        # .ravel() aplana la matriz 2D (filas × columnas) a una sola fila,
        # y con tuple unpacking se asigna cada valor a su componente
        TN, FP, FN, TP = matriz_conf.ravel()

        # Nº de casos de cada tipo × su impacto unitario, todo sumado
        VE = (TN * ITN) + (FP * IFP) + (FN * IFN) + (TP * ITP)
        return VE

    #CREAMOS UNA LISTA PARA EL VALOR ESPERADO
    ve_list = []

    #ITERAMOS CADA PUNTO DE CORTE Y RECOGEMOS SU VE
    # El paso de 0.01 define la SENSIBILIDAD: 100 cortes, uno cada 1%
    for umbral in np.arange(0, 1, 0.01):
        predicho = np.where(scoring > umbral, 1, 0)
        conf = confusion_matrix(real, predicho)
        ve_temp = valor_esperado(conf)
        ve_list.append(tuple([umbral, ve_temp]))

    #DEVUELVE EL RESULTADO COMO GRAFICO O COMO EL UMBRAL ÓPTIMO
    df_temp = pd.DataFrame(ve_list, columns=['umbral', 'valor_esperado'])

    if salida == 'grafico':
        solo_ve_positivo = df_temp[df_temp.valor_esperado > 0]
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=solo_ve_positivo, x='umbral', y='valor_esperado')
        plt.xticks(solo_ve_positivo.umbral, fontsize=14)
        plt.yticks(solo_ve_positivo.valor_esperado, fontsize=12);
    else:
        return df_temp.iloc[df_temp.valor_esperado.idxmax(), 0]


def pareto_ds4b(variable, salida='tabla'):
    pareto = variable.sort_values(ascending=False).to_frame()
    pareto.columns = ['Valor']
    pareto['Posicion']      = np.arange(start=1, stop=len(pareto) + 1)
    pareto['Posicion_Porc'] = pareto['Posicion'] / pareto.shape[0] * 100
    pareto['Acum']          = pareto['Valor'].cumsum()
    max_acum                = pareto['Acum'].max()
    pareto['Acum_Porc']     = pareto['Acum'] / max_acum * 100
    pareto              = pareto[['Posicion_Porc', 'Acum_Porc']]

    if salida == 'grafico':
        f, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pareto.Posicion_Porc, pareto.Acum_Porc)
        ax.plot(pareto.Posicion_Porc, pareto.Posicion_Porc)
        ax.tick_params(axis='x', labelsize=12, labelrotation=90)
        return ax
    else:
        return pareto


def construir_minicubo(df: pd.DataFrame,
                        dimensiones: list,
                        metrica: str) -> pd.DataFrame:
    return (df[dimensiones + [metrica]]
              .melt(id_vars=metrica)
              .groupby(['variable','value'])[metrica]
              .agg(conteo='count', media='mean'))


def graficar_minicubo(cubo: pd.DataFrame, top_n: int = 10):
    dims = cubo.index.get_level_values('variable').unique()
    for dim in dims:
        data = cubo.loc[dim].sort_values('media', ascending=False).head(top_n)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        data[['conteo']].plot.bar(ax=axes[0], color='steelblue',
                                   title=f'{dim} — Conteo')
        data[['media']].plot.bar(ax=axes[1],  color='tomato',
                                  title=f'{dim} — Lesividad Media')
        for ax in axes:
            ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        plt.tight_layout(); plt.show()


def scorecard(cubo_df: pd.DataFrame,
              escenario: list,
              col_valor: str = 'value',
              col_media: str = 'media') -> float:
    """
    Estima la métrica esperada para un escenario nunca visto en el histórico.
    El cubo debe tener columnas: variable, value, media (u otras si se especifican).
    """
    resultado = cubo_df.loc[cubo_df[col_valor].isin(escenario), col_media].mean()
    return resultado



# ────────────────────────────────────────────────────────────────────
# 9  Estadística y Muestreo
# ────────────────────────────────────────────────────────────────────


def tamaño_muestral_proporcion(z, me):
    """
    z  = desviaciones típicas del NC elegido (ej: 1.96 para NC=95%)
    de 100 veces que lo hicimos salga en 95 veces
    me = margen de error/desvio máximo en porcentaje (ej: 3 para 3%),
    Responde a: ¿qué % de la población hace X?
    
    """
    return ((z**2) * (50 * 50)) / (me**2)


def tamaño_muestral_media(z, sigma, me):
    """
    z     = desviaciones típicas del NC elegido (ej: 1.96 para NC=95%)
    sigma = desviación típica estimada de la variable
    me    = margen de error máximo en las unidades de la variable (€, kg, días...)
    Responde a: ¿cuánto vale en promedio X?
    """
    return (z**2 * sigma**2) / (me**2)



# ────────────────────────────────────────────────────────────────────
# 10  Ingeniería y ETL
# ────────────────────────────────────────────────────────────────────


def log(mensaje: str, nivel: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(CONFIG["formato_log"].format(ts=ts, nivel=nivel, msg=mensaje))


def get_db_connection():
    """
    Context manager para conexiones a PostgreSQL.
    Garantiza que la conexión se cierra siempre.
    """
    conexion = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    try:
        yield conexion
        conexion.commit()
    except Exception:
        conexion.rollback()
        raise
    finally:
        conexion.close()


def crear_session_robusta() -> requests.Session:
    """Session con reintentos automáticos ante errores de red."""
    session = requests.Session()

    reintentos = Retry(
        total=3,              # máximo 3 intentos
        backoff_factor=1,     # espera: 1s, 2s, 4s entre intentos
        status_forcelist=[429, 500, 502, 503, 504]   # errores que reintentar
    )

    adapter = HTTPAdapter(max_retries=reintentos)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def leer_carpeta_csv(ruta_carpeta: str) -> pl.DataFrame:
    """Lee todos los CSV de una carpeta y los apila"""
    ruta = Path(ruta_carpeta)
    archivos = sorted(ruta.glob("*.csv"))    # sorted = orden cronológico

    if not archivos:
        raise FileNotFoundError(f"No hay CSV en {ruta}")

    dfs = []
    for archivo in archivos:
        df = pl.read_csv(archivo)
        df = df.with_columns(
            pl.lit(archivo.stem).alias("fuente")    # agrega el nombre del archivo
        )
        dfs.append(df)

    return pl.concat(dfs)



# ────────────────────────────────────────────────────────────────────
# 11  Pipelines y Preproducción
# ────────────────────────────────────────────────────────────────────


def prepara_datos(df):
    """
    Pasos de FILAS, previos al pipeline.
    Corresponde al TIPO = P de la matriz de procesos x variables.
    Todo lo que transforme VALORES debe ir dentro del pipe, no aquí.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=COLUMNAS_OBLIGATORIAS)
    return df


def a_transformer(funcion, uno_a_uno=True):
    """
    funcion   = función que recibe una columna/DataFrame y devuelve lo transformado.
                NO puede cambiar el número de filas.
    uno_a_uno = True si la función devuelve tantas columnas como recibe.
                Necesario para que get_feature_names_out() no reviente.
    """
    if uno_a_uno:
        return FunctionTransformer(funcion, feature_names_out='one-to-one')
    return FunctionTransformer(funcion)


def construir_preprocesador(bloques, remainder='passthrough'):
    """
    bloques   = lista de tuplas (transformer, [columnas])
    remainder = 'passthrough' conserva las no nombradas (van al FINAL)
                'drop' LAS BORRA — es el defecto de sklearn, casi nunca lo que quieres

    Aborta si una columna aparece en dos bloques: el ColumnTransformer
    NO encadena, aplica cada bloque por separado sobre el valor original
    y la columna saldría dos veces.
    """
    vistas, repetidas = set(), []
    for transformer, columnas in bloques:
        for columna in columnas:
            if columna in vistas:
                repetidas.append(columna)
            else:
                vistas.add(columna)

    if repetidas:
        raise ValueError(
            f"Columnas en más de un bloque: {sorted(set(repetidas))}. "
            "Para encadenar procesos sobre la misma columna, "
            "usa make_pipeline(t1, t2) dentro de un único bloque."
        )

    return make_column_transformer(*bloques, remainder=remainder)


def guardar_pipeline(pipe, ruta):
    """
    Usa cloudpickle en lugar de pickle para que las funciones metidas
    con FunctionTransformer viajen DENTRO del artefacto.
    Coste medido: ~400 bytes extra. Ahorro: no arrastrar el código
    de las funciones a cada script de producción.
    """
    with open(ruta, mode='wb') as archivo:
        cloudpickle.dump(pipe, archivo)
    return ruta


def cargar_pipeline(ruta):
    with open(ruta, mode='rb') as archivo:
        pipe = cloudpickle.load(archivo)
    return pipe


def inspeccionar_pipeline(pipe):
    """
    Primer sitio donde mirar cuando un pipeline hace algo raro.
    Devuelve un dict con los pasos, el detalle del ColumnTransformer
    y los nombres de las columnas de salida.
    """
    informe = {}

    pasos = []
    for nombre in pipe.named_steps:
        pasos.append(nombre)
    informe['pasos'] = pasos

    if 'columntransformer' in pipe.named_steps:
        ct = pipe.named_steps['columntransformer']
        bloques = []
        for nombre, transformer, columnas in ct.transformers_:
            bloques.append({'bloque': nombre,
                            'transformer': type(transformer).__name__,
                            'columnas': columnas})
        informe['column_transformer'] = bloques
        informe['remainder'] = ct.remainder

        try:
            informe['columnas_salida'] = list(ct.get_feature_names_out())
        except AttributeError:
            informe['columnas_salida'] = (
                "No disponible: algún FunctionTransformer no declara "
                "feature_names_out='one-to-one'"
            )

    return informe
