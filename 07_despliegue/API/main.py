import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import RegistroEntrada, ScoringSalida
from .scoring import scoring_df

# ── Archivos y columnas ─────────
DIRECTORIO_API = Path(__file__).resolve().parent
RUTA_HISTORIAL = DIRECTORIO_API / "demo_clientes.csv"
RUTA_PAYLOAD_PRUEBA = DIRECTORIO_API / "test_payload.json"
COLUMNAS_HISTORIAL = [
    "ingresos",
    "ingresos_verificados",
    "rating",
    "dti",
    "antigüedad_empleo",
    "num_lineas_credito",
    "porc_uso_revolving",
    "num_derogatorios",
]

app = FastAPI(
    title="API de scoring de riesgos",
    description="Calcula PD, EAD, LGD y pérdida esperada para un préstamo.",
    version="1.0.0",
)


# ── Preparación de datos ─────────
def calcula_cuota(principal, tipo_interes, num_cuotas):
    tasa_mensual = tipo_interes / 100 / 12

    if tasa_mensual == 0:
        return principal / num_cuotas

    return principal * tasa_mensual / (1 - (1 + tasa_mensual) ** -num_cuotas)


def prepara_datos(registros):
    if not RUTA_HISTORIAL.exists():
        raise HTTPException(status_code=500, detail="No se encontró demo_clientes.csv.")

    historial = pd.read_csv(RUTA_HISTORIAL)
    columnas_requeridas = ["id_cliente", *COLUMNAS_HISTORIAL]

    for columna in columnas_requeridas:
        if columna not in historial.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Falta la columna {columna} en demo_clientes.csv.",
            )

    if historial["id_cliente"].duplicated().any():
        raise HTTPException(status_code=500, detail="Hay id_cliente repetidos en el historial.")

    historial = historial.set_index("id_cliente")
    filas = []

    for registro in registros:
        if registro.id_cliente not in historial.index:
            raise HTTPException(
                status_code=404,
                detail=f"No hay historial para id_cliente {registro.id_cliente}.",
            )

        fila = registro.model_dump(by_alias=True)
        cliente = historial.loc[registro.id_cliente]

        for columna in COLUMNAS_HISTORIAL:
            fila[columna] = cliente[columna]

        fila["num_cuotas"] = f"{registro.num_cuotas} months"
        fila["imp_cuota"] = calcula_cuota(
            registro.principal,
            registro.tipo_interes,
            registro.num_cuotas,
        )
        filas.append(fila)

    return pd.DataFrame(filas)


# ── Scoring ─────────
def puntuar(registros):
    datos = prepara_datos(registros)
    return datos, scoring_df(datos)


def a_registros_json(resultados):
    return json.loads(resultados.to_json(orient="records"))


# ── Endpoints HTTP ─────────
def health():
    return {"status": "OK"}


def predict(registros: list[RegistroEntrada]):
    try:
        _, resultados = puntuar(registros)
        return a_registros_json(resultados)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


def debug():
    informe = {
        "python_version": sys.version,
        "status": "ERROR",
        "input_columns_detected": [],
        "output_columns_detected": [],
        "sample_output": [],
        "engine_error": None,
    }

    try:
        payload = json.loads(RUTA_PAYLOAD_PRUEBA.read_text(encoding="utf-8"))
        registros = []

        for fila in payload:
            registros.append(RegistroEntrada.model_validate(fila))

        datos, resultados = puntuar(registros)
        informe["status"] = "OK"
        informe["input_columns_detected"] = list(datos.columns)
        informe["output_columns_detected"] = list(resultados.columns)
        informe["sample_output"] = a_registros_json(resultados.head(1))
    except Exception as error:  # noqa: BLE001
        informe["engine_error"] = str(error)

    return informe


app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/predict", predict, methods=["POST"], response_model=list[ScoringSalida])
app.add_api_route("/debug", debug, methods=["GET"])
