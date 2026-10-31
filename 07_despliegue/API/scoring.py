from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd


# ── Rutas y variables del modelo ─────────
DIRECTORIO_API = Path(__file__).resolve().parent
DIRECTORIO_DESPLIEGUE = DIRECTORIO_API.parent
ID = "id_cliente"
COLUMNAS_PREDICTORAS = [
    "ingresos_verificados",
    "vivienda",
    "finalidad",
    "num_cuotas",
    "antigüedad_empleo",
    "rating",
    "ingresos",
    "dti",
    "num_lineas_credito",
    "porc_uso_revolving",
    "principal",
    "tipo_interes",
    "imp_cuota",
    "num_derogatorios",
]
NOMBRES_MODELOS = [
    "artefacto_pipeline_pd.pkl",
    "artefacto_pipeline_ead.pkl",
    "artefacto_pipeline_lgd.pkl",
]
MODELOS = None


# ── Carga de los tres modelos ─────────
def cargar_modelos():
    global MODELOS #interesante

    if MODELOS is None:
        modelos_cargados = []

        for nombre in NOMBRES_MODELOS:
            ruta_api = DIRECTORIO_API / nombre
            ruta_despliegue = DIRECTORIO_DESPLIEGUE / nombre

            if ruta_api.exists():
                ruta_artefacto = ruta_api
            else:
                ruta_artefacto = ruta_despliegue

            with open(ruta_artefacto, "rb") as archivo:
                modelos_cargados.append(cloudpickle.load(archivo))

        MODELOS = modelos_cargados

    return MODELOS


# ── Predicción en memoria ─────────
def scoring_df(df):
    pipeline_pd, pipeline_ead, pipeline_lgd = cargar_modelos()

    X_scoring = df.loc[:, COLUMNAS_PREDICTORAS].copy()
    X_scoring["ingresos"] = X_scoring["ingresos"].clip(upper=400000)
    resultados = pd.DataFrame({ID: df[ID].to_numpy()})

    resultados["pd_score"] = pipeline_pd.predict_proba(X_scoring)[:, 1]
    resultados["ead_score"] = np.clip(pipeline_ead.predict(X_scoring), 0, 1)
    resultados["lgd_score"] = np.clip(pipeline_lgd.predict(X_scoring), 0, 1)
    resultados["perdida_esperada"] = (
        resultados["pd_score"]
        * df["principal"].to_numpy()
        * resultados["ead_score"]
        * resultados["lgd_score"]
    )

    return resultados
