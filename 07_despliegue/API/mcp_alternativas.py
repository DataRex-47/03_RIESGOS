# mcp_alternativas.py (lado API): reglas de las alternativas y servidor MCP
# Pieza nueva de la API. Guarda las reglas, las publica como tools para el asistente
# y protege la puerta con la misma clave X-API-Key. La usa chat_ia.py (en la app).
import os
import secrets
from typing import Literal

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from schemas import FINALIDAD, VIVIENDA, RegistroEntrada
from scoring import calcula_tasa_sugerida, scoring_df


# ── Paso 1. Reglas: cortes y límites ─────────
# Los cortes deben ser los mismos que UMBRAL_BAJO y UMBRAL_ALTO al inicio de app/app.py.
UMBRAL_BAJO = 0.08
UMBRAL_ALTO = 0.15

# Lo que el modelo conoce (los mismos límites de RegistroEntrada en schemas.py)
PLAZOS_PERMITIDOS = [36, 60]
MONTO_MINIMO = 500
MONTO_MAXIMO = 40000

# Los montos de las alternativas son múltiplos de este paso, y esto es lo que se prueba bajar:
# 10 % menos, 20 % menos... hasta 70 % menos
PASO_MONTO = 500
REDUCCIONES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# Tope de pérdida esperada (en dólares). None = solo cuenta la probabilidad de impago.
# Con un número (por ejemplo 2000), una alternativa que sigue en riesgo alto también sirve si
# su pérdida esperada queda bajo el tope. El asistente debe decir que sigue en riesgo alto.
TOPE_PERDIDA_ESPERADA = 2000


# ── Nivel de riesgo según la probabilidad de impago ─────────
def nivel_por_pd(pd_score):
    if pd_score < UMBRAL_BAJO:
        return "bajo"
    if pd_score <= UMBRAL_ALTO:
        return "medio"
    return "alto"


# ── Qué número tiene cada nivel (sirve para comparar niveles) ─────────
def numero_de_nivel(nivel):
    if nivel == "bajo":
        return 0
    if nivel == "medio":
        return 1
    return 2


# main.py llena esto al montar el MCP. Así este archivo no tiene que importar main.py
# (importarse el uno al otro se enredaría).
contexto = {"prepara_entrada": None}


# ── Paso 2. Funciones que calculan las alternativas ─────────
# No usan IA ni la librería MCP: son Python normal y se pueden probar solas.

# Un escenario es un par [monto, plazo]. Todos se evalúan JUNTOS, en una sola llamada
# al modelo, porque llamar uno por uno sería lento.
def evalua_escenarios(prepara_entrada, id_cliente, finalidad, vivienda, pares):
    solicitudes = []
    for par in pares:
        solicitud = RegistroEntrada(
            id_cliente=id_cliente,
            principal=par[0],
            num_cuotas=par[1],
            finalidad=finalidad,
            vivienda=vivienda,
        )
        solicitudes.append(solicitud)

    datos = prepara_entrada(solicitudes)
    resultados = scoring_df(datos)

    escenarios = []
    posicion = 0
    for _, fila in resultados.iterrows():
        # Las filas salen en el mismo orden que los pares
        monto = pares[posicion][0]
        plazo = pares[posicion][1]
        posicion = posicion + 1

        tasa = calcula_tasa_sugerida(fila["perdida_esperada"], monto, plazo)
        escenarios.append(
            {
                "principal": monto,
                "num_cuotas": plazo,
                "pd": round(float(fila["pd_score"]), 4),
                "ead": round(float(fila["ead_score"]), 4),
                "lgd": round(float(fila["lgd_score"]), 4),
                "perdida_esperada": round(float(fila["perdida_esperada"]), 2),
                "tasa_sugerida": round(float(tasa), 2),
                "nivel": nivel_por_pd(float(fila["pd_score"])),
            }
        )
    return escenarios


def redondea_abajo(monto):
    return int(monto // PASO_MONTO) * PASO_MONTO


# Escenarios que se van a probar: el monto original y versiones con 10 %, 20 %... menos,
# combinados con cada plazo permitido, sin repetir y sin incluir la solicitud original
def arma_candidatos(principal, num_cuotas):
    montos = [principal]
    for reduccion in REDUCCIONES:
        monto = redondea_abajo(principal * (1 - reduccion))
        if monto >= MONTO_MINIMO and monto < principal and monto not in montos:
            montos.append(monto)

    pares = []
    for monto in montos:
        for plazo in PLAZOS_PERMITIDOS:
            es_original = monto == principal and plazo == num_cuotas
            if not es_original:
                pares.append([monto, plazo])
    return pares


# ¿Sirve este escenario como alternativa? Siempre debe bajar la probabilidad de impago, y además:
# sale del riesgo alto, o (si hay tope) sigue en alto pero con la pérdida esperada bajo el tope
def sirve_como_alternativa(escenario, original):
    if escenario["pd"] >= original["pd"]:
        return False
    if escenario["nivel"] != "alto":
        return True
    if TOPE_PERDIDA_ESPERADA is not None and escenario["perdida_esperada"] <= TOPE_PERDIDA_ESPERADA:
        return True
    return False


# Puntaje para ordenar las alternativas: menor es mejor.
# puntaje = 0.5 * riesgo + 0.3 * cuánto cambia la solicitud + 0.2 * PD relativa al corte alto
def calcula_puntaje(escenario, principal, num_cuotas):
    riesgo = numero_de_nivel(escenario["nivel"])
    reduccion_monto = (principal - escenario["principal"]) / principal
    cambio_plazo = abs(escenario["num_cuotas"] - num_cuotas) / num_cuotas
    impacto = reduccion_monto + cambio_plazo
    pd_relativa = escenario["pd"] / UMBRAL_ALTO
    return round(0.5 * riesgo + 0.3 * impacto + 0.2 * pd_relativa, 3)


def clave_puntaje(alternativa):
    return alternativa["puntaje"]


def dice_motivo(escenario, principal, num_cuotas):
    cambio_monto = escenario["principal"] != principal
    cambio_plazo = escenario["num_cuotas"] != num_cuotas
    if cambio_monto and cambio_plazo:
        return "monto y plazo"
    if cambio_monto:
        return "monto menor"
    return "plazo distinto"


# La función principal: propone alternativas para una solicitud de riesgo alto
def propone_alternativas(prepara_entrada, id_cliente, principal, num_cuotas, finalidad, vivienda, maximo):
    # Primero, el escenario original
    original = evalua_escenarios(prepara_entrada, id_cliente, finalidad, vivienda, [[principal, num_cuotas]])[0]

    # Solo se buscan alternativas si el original es de riesgo alto
    if original["nivel"] != "alto":
        return {
            "original": original,
            "alternativas": [],
            "sin_alternativas": True,
            "motivo_sin_alternativas": "El escenario original no es de riesgo alto.",
        }

    candidatos = arma_candidatos(principal, num_cuotas)
    if candidatos == []:
        return {
            "original": original,
            "alternativas": [],
            "sin_alternativas": True,
            "motivo_sin_alternativas": "No hay variaciones válidas de monto ni de plazo.",
        }

    # Se evalúan todos los candidatos juntos y se queda con los que sirven
    evaluados = evalua_escenarios(prepara_entrada, id_cliente, finalidad, vivienda, candidatos)
    alternativas = []
    for escenario in evaluados:
        if sirve_como_alternativa(escenario, original):
            escenario["motivo"] = dice_motivo(escenario, principal, num_cuotas)
            escenario["puntaje"] = calcula_puntaje(escenario, principal, num_cuotas)
            escenario["sigue_en_riesgo_alto"] = escenario["nivel"] == "alto"
            alternativas.append(escenario)

    if alternativas == []:
        return {
            "original": original,
            "alternativas": [],
            "sin_alternativas": True,
            "motivo_sin_alternativas": "Ninguna variación de monto o plazo sirve: ninguna baja el riesgo por debajo del corte alto ni deja la pérdida esperada bajo el tope.",
        }

    # Orden: el menor puntaje primero; solo se devuelven las mejores
    alternativas = sorted(alternativas, key=clave_puntaje)
    alternativas = alternativas[:maximo]
    for alternativa in alternativas:
        alternativa["es_mejor"] = False
    alternativas[0]["es_mejor"] = True

    return {
        "original": original,
        "alternativas": alternativas,
        "sin_alternativas": False,
        "motivo_sin_alternativas": "",
    }


# Para cada plazo, el mayor monto (múltiplo de 500) cuyo nivel no supera el nivel objetivo
def calcula_monto_maximo(prepara_entrada, id_cliente, finalidad, vivienda, nivel_objetivo):
    limite = numero_de_nivel(nivel_objetivo)

    # Todos los montos posibles, del más alto al más bajo, para cada plazo
    pares = []
    for plazo in PLAZOS_PERMITIDOS:
        monto = MONTO_MAXIMO
        while monto >= MONTO_MINIMO:
            pares.append([monto, plazo])
            monto = monto - PASO_MONTO

    evaluados = evalua_escenarios(prepara_entrada, id_cliente, finalidad, vivienda, pares)

    # Por plazo, el primero que cumple (como van de mayor a menor, es el máximo)
    respuesta = []
    for plazo in PLAZOS_PERMITIDOS:
        encontrado = None
        for escenario in evaluados:
            if encontrado is None and escenario["num_cuotas"] == plazo:
                if numero_de_nivel(escenario["nivel"]) <= limite:
                    encontrado = escenario
        if encontrado is None:
            respuesta.append(
                {"num_cuotas": plazo, "hay_monto": False, "detalle": "Ningún monto de 500 a 40000 cumple ese nivel."}
            )
        else:
            encontrado["hay_monto"] = True
            respuesta.append(encontrado)
    return {"nivel_objetivo": nivel_objetivo, "por_plazo": respuesta}


# ── Paso 3. Servidor MCP y sus cuatro tools ─────────
mcp = FastMCP("Scoring de riesgos")


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def consultar_politica() -> dict:
    """Devuelve las reglas del banco: los cortes del semáforo de riesgo (según la
    probabilidad de impago), los plazos permitidos, el rango de monto y qué se puede
    cambiar en una solicitud. Úsala cuando necesites citar una regla; nunca la inventes."""
    return {
        "corte_bajo": UMBRAL_BAJO,
        "corte_alto": UMBRAL_ALTO,
        "reglas_de_nivel": "bajo si la PD es menor al corte bajo; medio si está entre ambos cortes; alto si supera el corte alto",
        "plazos_permitidos_meses": PLAZOS_PERMITIDOS,
        "monto_minimo": MONTO_MINIMO,
        "monto_maximo": MONTO_MAXIMO,
        "paso_del_monto": PASO_MONTO,
        "tope_perdida_esperada": TOPE_PERDIDA_ESPERADA,
        "se_puede_cambiar": ["monto", "plazo"],
        "no_se_puede_cambiar": ["cliente", "finalidad", "vivienda", "datos del cliente"],
    }


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def evaluar_escenario(
    id_cliente: int,
    principal: float,
    num_cuotas: Literal[36, 60],
    finalidad: FINALIDAD,
    vivienda: VIVIENDA,
) -> dict:
    """Calcula con los modelos reales el riesgo de UN escenario de préstamo: probabilidad
    de impago, pérdida esperada, tasa sugerida y nivel (bajo, medio o alto). El monto va de
    500 a 40000. Úsala cuando pregunten qué pasaría con un monto y un plazo concretos."""
    try:
        pares = [[principal, num_cuotas]]
        return evalua_escenarios(contexto["prepara_entrada"], id_cliente, finalidad, vivienda, pares)[0]
    except Exception:
        return {"error": "No pude evaluar ese escenario. Revisa que el cliente exista y que el monto esté entre 500 y 40000."}


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def proponer_alternativas(
    id_cliente: int,
    principal: float,
    num_cuotas: Literal[36, 60],
    finalidad: FINALIDAD,
    vivienda: VIVIENDA,
    maximo: int = 3,
) -> dict:
    """Dada una solicitud de riesgo alto, genera variaciones de monto y plazo, las evalúa con
    los modelos reales, descarta las que no sirven y ordena el resto de mejor a peor. Devuelve
    la solicitud original y las alternativas (la primera es la mejor); cada una dice si sigue
    en riesgo alto. Si no hay ninguna, lo indica y dice por qué. El monto va de 500 a 40000 y
    maximo (de 1 a 5) es cuántas alternativas devolver. Úsala primero cuando pidan opciones."""
    if maximo < 1:
        maximo = 1
    if maximo > 5:
        maximo = 5
    try:
        return propone_alternativas(
            contexto["prepara_entrada"], id_cliente, principal, num_cuotas, finalidad, vivienda, maximo
        )
    except Exception:
        return {"error": "No pude calcular alternativas. Revisa que el cliente exista y que el monto esté entre 500 y 40000."}


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def monto_maximo(
    id_cliente: int,
    finalidad: FINALIDAD,
    vivienda: VIVIENDA,
    nivel_objetivo: Literal["bajo", "medio"] = "medio",
) -> dict:
    """Calcula, para cada plazo permitido, el mayor monto (múltiplo de 500) cuyo nivel de
    riesgo no supera el nivel objetivo. Úsala cuando pregunten cuánto se podría prestar."""
    try:
        return calcula_monto_maximo(contexto["prepara_entrada"], id_cliente, finalidad, vivienda, nivel_objetivo)
    except Exception:
        return {"error": "No pude calcular el monto máximo. Revisa que el cliente exista."}


# ── Paso 4. Clave de acceso ─────────
# Mismo control que verifica_clave de seguridad.py: la cabecera X-API-Key debe
# coincidir con la variable de entorno API_KEY. Sin API_KEY, queda abierto (como el resto).
class ExigeClave(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        esperada = os.environ.get("API_KEY")
        if esperada:
            clave = request.headers.get("X-API-Key")
            if clave is None or not secrets.compare_digest(clave.encode(), esperada.encode()):
                return JSONResponse({"detail": "Clave inválida o ausente"}, status_code=401)
        return await call_next(request)


# ── Paso 5. Aplicación que main.py monta en /mcp ─────────
def crea_app_mcp(prepara_entrada):
    contexto["prepara_entrada"] = prepara_entrada
    return mcp.http_app(
        path="/",
        middleware=[Middleware(ExigeClave)],
        stateless_http=True,
        json_response=True,
    )
