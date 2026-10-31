from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import RegistroEntrada, ScoringSalida
from scoring import calcula_tasa_sugerida, scoring_df

# ── Historial de clientes ─────────
# Este CSV simula la base de datos del banco: el historial de cada cliente.
# En producción sería una consulta a una base de datos protegida.
CARPETA = Path(__file__).resolve().parent

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

historial = pd.read_csv(CARPETA / "demo_clientes.csv").set_index("id_cliente")


# ── Entrada del modelo: solicitud + historial ─────────
def prepara_entrada(solicitudes):
    filas = []
    for solicitud in solicitudes:
        if solicitud.id_cliente not in historial.index:
            raise HTTPException(
                status_code=404,
                detail=f"No hay historial para el cliente {solicitud.id_cliente}",
            )

        # Los cinco datos de la solicitud + los ocho del historial
        fila = solicitud.model_dump()
        fila.update(historial.loc[solicitud.id_cliente, COLUMNAS_HISTORIAL].to_dict())

        # El modelo aprendió el plazo como texto con un espacio delante: " 36 months"
        fila["num_cuotas"] = " " + str(solicitud.num_cuotas) + " months"
        filas.append(fila)

    return pd.DataFrame(filas)


# ── Aplicación ─────────
app = FastAPI(title="API de scoring de riesgos")


# ── Rutas ─────────
@app.get("/health")
def health():
    return {"status": "OK"}


# La app llama aquí para llenar su selector de clientes
@app.get("/clientes")
def clientes():
    # Solo enviamos los ids, nunca el historial del cliente
    ids = []
    for id_cliente in historial.index:
        ids.append(int(id_cliente))
    return {"ids": ids}


# Ficha del cliente: lo que el banco ya sabe de él (en la demo, datos ficticios)
@app.get("/clientes/{id_cliente}")
def ficha_cliente(id_cliente: int):
    if id_cliente not in historial.index:
        raise HTTPException(
            status_code=404,
            detail=f"No hay historial para el cliente {id_cliente}",
        )

    datos = historial.loc[id_cliente]
    return {
        "id_cliente": id_cliente,
        "ingresos": float(datos["ingresos"]),
        "ingresos_verificados": str(datos["ingresos_verificados"]),
        "rating": str(datos["rating"]),
        "dti": float(datos["dti"]),
        "antiguedad_empleo": str(datos["antigüedad_empleo"]),
        "num_lineas_credito": int(datos["num_lineas_credito"]),
        "porc_uso_revolving": float(datos["porc_uso_revolving"]),
        "num_derogatorios": int(datos["num_derogatorios"]),
    }


@app.post("/predict", response_model=list[ScoringSalida])
def predict(solicitudes: list[RegistroEntrada]):
    datos = prepara_entrada(solicitudes)

    try:
        resultados = scoring_df(datos)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error en el modelo: {error}")

    # Pasamos cada fila a tipos simples de Python para la respuesta
    respuesta = []
    posicion = 0
    for _, fila in resultados.iterrows():
        # Las filas de salida van en el mismo orden que las solicitudes
        solicitud = solicitudes[posicion]
        posicion = posicion + 1
        tasa_sugerida = calcula_tasa_sugerida(
            fila["perdida_esperada"], solicitud.principal, solicitud.num_cuotas
        )
        respuesta.append(
            {
                "id_cliente": int(fila["id_cliente"]),
                "pd_score": float(fila["pd_score"]),
                "ead_score": float(fila["ead_score"]),
                "lgd_score": float(fila["lgd_score"]),
                "perdida_esperada": float(fila["perdida_esperada"]),
                "tasa_sugerida": float(tasa_sugerida),
            }
        )
    return respuesta
