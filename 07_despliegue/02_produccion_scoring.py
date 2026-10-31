import argparse
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

# ── Rutas y columnas ─────────
RAIZ = Path(__file__).resolve().parents[1]
DIRECTORIO_ARTEFACTOS = RAIZ / "07_despliegue"
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

# ── Tope de ingresos ─────────
# En el entrenamiento se descartaron los préstamos con ingresos superiores a este valor.
# En producción no se descarta a nadie: un ingreso mayor se puntúa como si fuera el tope,
# para que el escalado del modelo no reciba valores fuera del rango que conoce.
TOPE_INGRESOS = 400000


# ── Entrada de scoring ─────────
parser = argparse.ArgumentParser(description="Genera predicciones PD, EAD y LGD.")
parser.add_argument("--input", required=True, type=Path, help="CSV de préstamos nuevos")
parser.add_argument("--output", required=True, type=Path, help="CSV donde guardar el scoring")
args = parser.parse_args()

datos = pd.read_csv(args.input, sep=",", encoding="utf-8")
X_scoring = datos.loc[:, COLUMNAS_PREDICTORAS].copy()
X_scoring["ingresos"] = X_scoring["ingresos"].clip(upper=TOPE_INGRESOS)
resultados = pd.DataFrame({ID: datos[ID].to_numpy()})

# ── Carga de modelos y predicción ─────────
with open(DIRECTORIO_ARTEFACTOS / "artefacto_pipeline_pd.pkl", "rb") as archivo_artefacto:
    pipeline_pd = cloudpickle.load(archivo_artefacto)
resultados["pd_score"] = pipeline_pd.predict_proba(X_scoring)[:, 1]

with open(DIRECTORIO_ARTEFACTOS / "artefacto_pipeline_ead.pkl", "rb") as archivo_artefacto:
    pipeline_ead = cloudpickle.load(archivo_artefacto)
resultados["ead_score"] = np.clip(pipeline_ead.predict(X_scoring), 0, 1)

with open(DIRECTORIO_ARTEFACTOS / "artefacto_pipeline_lgd.pkl", "rb") as archivo_artefacto:
    pipeline_lgd = cloudpickle.load(archivo_artefacto)
resultados["lgd_score"] = np.clip(pipeline_lgd.predict(X_scoring), 0, 1)

# ── Pérdida esperada ─────────
# Pérdida esperada = PD x principal x EAD x LGD, como indica el manifiesto de preproducción.
resultados["perdida_esperada"] = (
    resultados["pd_score"]
    * datos["principal"].to_numpy()
    * resultados["ead_score"]
    * resultados["lgd_score"]
)

# ── Guardado de resultados ─────────
args.output.parent.mkdir(parents=True, exist_ok=True)
resultados.to_csv(args.output, index=False, encoding="utf-8")
print(f"Scoring guardado: {args.output} ({len(resultados)} préstamos)")
