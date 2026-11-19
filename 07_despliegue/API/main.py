import unicodedata
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

# Contratos de entrada y salida (schemas), motor de cálculo (scoring), clave y MCP
from schemas import RegistroEntrada, ScoringSalida
from scoring import calcula_tasa_sugerida, scoring_df
from seguridad import verifica_clave
from mcp_alternativas import crea_app_mcp

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


# ── Búsqueda por nombre ─────────
# El CSV trae el nombre de cada cliente (inventado). Para buscar sin que una tilde
# estorbe, guardamos una versión del nombre en minúsculas y sin tildes: "María" -> "maria"
def sin_tildes(texto):
    texto = unicodedata.normalize("NFD", texto.lower())
    letras = []
    for letra in texto:
        # Las tildes quedan como marcas aparte (categoría "Mn"); esas se descartan
        if unicodedata.category(letra) != "Mn":
            letras.append(letra)
    return "".join(letras)


historial["nombre_busqueda"] = historial["nombre"].apply(sin_tildes)


# ── Entrada del modelo: solicitud + historial ─────────
def prepara_entrada(solicitudes):
    filas = []

    for solicitud in solicitudes:
        # Si el cliente no está en el historial, no hay con qué calcular
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
# El MCP es un servidor aparte que vive dentro de esta misma API: sirve las tools del chat.
mcp_app = crea_app_mcp(prepara_entrada)
app = FastAPI(title="API de scoring de riesgos", lifespan=mcp_app.lifespan)
app.mount("/mcp", mcp_app)


# ── Rutas ─────────
@app.get("/health")
def health():
    return {"status": "OK"}


# Lista completa de ids. Es solo de demo: con 200.000 clientes sería inviable,
# por eso existe la búsqueda de abajo.
# dependencies=[Depends(verifica_clave)] exige la clave X-API-Key (si hay una configurada)
@app.get("/clientes", dependencies=[Depends(verifica_clave)])
def clientes():
    # Solo enviamos los ids, nunca el historial del cliente
    ids = []
    for id_cliente in historial.index:
        ids.append(int(id_cliente))
    return {"ids": ids}


# Búsqueda por nombre, paginada: la app escribe unas letras y recibe pocos resultados.
# Va ANTES de "/clientes/{id_cliente}", si no FastAPI leería "buscar" como un id.
@app.get("/clientes/buscar", dependencies=[Depends(verifica_clave)])
def busca_clientes(
    q: str = Query("", max_length=50),
    pagina: int = Query(1, ge=1),
    tamano: int = Query(10, ge=1, le=50),
):
    # Se queda con los clientes cuyo nombre contiene el texto buscado
    coincidencias = historial[
        historial["nombre_busqueda"].str.contains(sin_tildes(q.strip()), regex=False)
    ]

    # Paginación: se devuelve solo el trozo pedido
    inicio = (pagina - 1) * tamano
    trozo = coincidencias.iloc[inicio : inicio + tamano]

    resultados = []
    for id_cliente, fila in trozo.iterrows():
        resultados.append({"id_cliente": int(id_cliente), "nombre": str(fila["nombre"])})
    return {
        "total": int(len(coincidencias)),
        "pagina": pagina,
        "tamano": tamano,
        "resultados": resultados,
    }


# Ficha del cliente: lo que el banco ya sabe de él (en la demo, datos de una base pública)
@app.get("/clientes/{id_cliente}", dependencies=[Depends(verifica_clave)])
def ficha_cliente(id_cliente: int):
    if id_cliente not in historial.index:
        raise HTTPException(
            status_code=404,
            detail=f"No hay historial para el cliente {id_cliente}",
        )

    datos = historial.loc[id_cliente]
    return {
        "id_cliente": id_cliente,
        "nombre": str(datos["nombre"]),
        "ingresos": float(datos["ingresos"]),
        "ingresos_verificados": str(datos["ingresos_verificados"]),
        "rating": str(datos["rating"]),
        "dti": float(datos["dti"]),
        "antiguedad_empleo": str(datos["antigüedad_empleo"]),
        "num_lineas_credito": int(datos["num_lineas_credito"]),
        "porc_uso_revolving": float(datos["porc_uso_revolving"]),
        "num_derogatorios": int(datos["num_derogatorios"]),
    }

# ── Ruta principal: calcula el scoring de una o varias solicitudes ─────────
@app.post("/predict", response_model=list[ScoringSalida], dependencies=[Depends(verifica_clave)])
def predict(solicitudes: list[RegistroEntrada]):
    # ── Paso 1. De la solicitud a la tabla del modelo ─────────
    datos = prepara_entrada(solicitudes)

    # ── Paso 2. Los tres modelos calculan ─────────
    try:
        resultados = scoring_df(datos)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error en el modelo: {error}")

    # ── Paso 3. De la tabla de resultados a una lista de diccionarios ─────────
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

    # ── Paso 4. FastAPI convierte esta lista en el JSON de respuesta ─────────
    return respuesta
