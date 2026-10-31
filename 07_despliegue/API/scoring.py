from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

# ── Columnas que esperan los modelos ─────────
# Las mismas que usa 02_produccion_scoring.py, en el mismo orden.
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

# En el entrenamiento se descartaron los ingresos mayores a este valor
TOPE_INGRESOS = 400000

# ── Carga de los tres modelos (una sola vez, al arrancar) ─────────
CARPETA = Path(__file__).resolve().parent


def carga_modelo(nombre_archivo):
    with open(CARPETA / nombre_archivo, "rb") as archivo:
        return cloudpickle.load(archivo)


MODELO_PD = carga_modelo("artefacto_pipeline_pd.pkl")
MODELO_EAD = carga_modelo("artefacto_pipeline_ead.pkl")
MODELO_LGD = carga_modelo("artefacto_pipeline_lgd.pkl")


# ── Scoring ─────────
def scoring_df(datos):
    # Solo las columnas del modelo (el id se queda fuera)
    X = datos[COLUMNAS_PREDICTORAS].copy()
    X["ingresos"] = X["ingresos"].clip(upper=TOPE_INGRESOS)

    resultados = pd.DataFrame({"id_cliente": datos["id_cliente"]})

    # PD: probabilidad de la clase 1 (impago)
    resultados["pd_score"] = MODELO_PD.predict_proba(X)[:, 1]

    # EAD y LGD son proporciones entre 0 y 1
    resultados["ead_score"] = np.clip(MODELO_EAD.predict(X), 0, 1)
    resultados["lgd_score"] = np.clip(MODELO_LGD.predict(X), 0, 1)

    # Pérdida esperada = PD x principal x EAD x LGD
    resultados["perdida_esperada"] = (
        resultados["pd_score"]
        * datos["principal"]
        * resultados["ead_score"]
        * resultados["lgd_score"]
    )
    return resultados
