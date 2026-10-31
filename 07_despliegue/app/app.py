# ─────────────────────────────────────────────
# Bancandrex: Mesa de decisión de crédito
# La app recoge datos, llama a la API y dibuja la respuesta.
# No tiene Machine Learning: todo el cálculo vive en la API.
# ─────────────────────────────────────────────

# ── Configuración (lo único que se edita) ─────────
# En Render se define la variable de entorno API_BASE_URL y no hay que tocar el código
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 60  # Segundos máximos para cada llamada
ESPERA_MAXIMA = 120  # Segundos máximos esperando que la API despierte

# Cortes del semáforo sobre la probabilidad de impago (PD)
UMBRAL_BAJO = 0.08
UMBRAL_ALTO = 0.15

# Colores de la marca y del semáforo
AZUL = "#12304A"
TURQUESA = "#137C86"
VERDE = "#2E9E6B"
AMBAR = "#E3A72F"
ROJO = "#C8453B"
GRIS = "#5B6670"

# Opciones del formulario: texto que ve el analista -> valor que acepta la API
PLAZOS = {"36 meses": 36, "60 meses": 60}
VIVIENDAS = {"Arriendo": "RENT", "Propia": "OWN", "Hipoteca": "MORTGAGE", "Otra": "OTHER"}
FINALIDADES = {
    "Consolidar deudas": "debt_consolidation",
    "Pagar tarjeta de crédito": "credit_card",
    "Mejoras del hogar": "home_improvement",
    "Compra grande": "major_purchase",
    "Pequeño negocio": "small_business",
    "Vehículo": "car",
    "Gastos médicos": "medical",
    "Mudanza": "moving",
    "Vacaciones": "vacation",
    "Vivienda": "house",
    "Boda": "wedding",
    "Educación": "educational",
    "Energía renovable": "renewable_energy",
    "Otra": "other",
}

import time
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

CARPETA = Path(__file__).resolve().parent


# ── Formato de números ─────────
def dinero(valor):
    return "$" + f"{valor:,.0f}"


def porcentaje(valor):
    return f"{valor * 100:.1f} %"


# ── Cliente HTTP ─────────
def despierta_api():
    # Pregunta a /health hasta que responda (Render duerme el servicio gratuito)
    inicio = time.time()
    while time.time() - inicio < ESPERA_MAXIMA:
        try:
            if requests.get(API_BASE_URL + "/health", timeout=10).status_code == 200:
                return True
        except requests.ConnectionError:
            return False  # La API está apagada: no hay nada que esperar
        except requests.Timeout:
            pass
        time.sleep(3)
    return False


def llama_api(metodo, ruta, datos=None):
    # Devuelve (respuesta, error). Si todo sale bien, error es None
    try:
        respuesta = requests.request(metodo, API_BASE_URL + ruta, json=datos, timeout=TIMEOUT)
    except requests.ConnectionError:
        return None, "No hay conexión con el servicio. Revisa que la API esté encendida."
    except requests.Timeout:
        return None, "El servicio tardó demasiado en responder. Inténtalo de nuevo."

    if respuesta.status_code == 404:
        return None, respuesta.json()["detail"]
    if respuesta.status_code == 422:
        return None, "Algún dato está fuera de rango. Revisa el monto o el plazo."
    if respuesta.status_code != 200:
        return None, "No pudimos calcular el riesgo. Inténtalo de nuevo en un momento."
    return respuesta.json(), None


def calcula(registros):
    # Cada cálculo despierta primero la API y luego envía los registros
    with st.spinner("Analizando la solicitud. Si el servicio estaba dormido, puede tardar hasta un minuto."):
        despierta_api()
        return llama_api("POST", "/predict", registros)


def carga_clientes():
    # La API solo manda los ids; se piden una vez y se guardan
    if "ids" not in st.session_state:
        with st.spinner("Conectando con el servicio. Si estaba dormido, puede tardar hasta un minuto."):
            despierta_api()
            datos, error = llama_api("GET", "/clientes")
        if error:
            st.error(error)
            st.stop()
        st.session_state["ids"] = datos["ids"]
    return st.session_state["ids"]


def carga_ficha(id_cliente):
    # La ficha es lo que el banco ya sabe del cliente; se pide una vez por cliente
    if "fichas" not in st.session_state:
        st.session_state["fichas"] = {}
    if id_cliente not in st.session_state["fichas"]:
        ficha, error = llama_api("GET", "/clientes/" + str(id_cliente))
        st.session_state["fichas"][id_cliente] = ficha  # Si falla queda en None
    return st.session_state["fichas"][id_cliente]


# ── Datos de la solicitud ─────────
def arma_registro(solicitud, num_cuotas):
    # Estos cinco campos son el contrato de entrada de la API (schemas.py).
    # La tasa no se envía: la fija el banco según el riesgo y la API la devuelve.
    return {
        "id_cliente": solicitud["id_cliente"],
        "principal": solicitud["principal"],
        "num_cuotas": num_cuotas,
        "finalidad": solicitud["finalidad"],
        "vivienda": solicitud["vivienda"],
    }


def nivel_riesgo(pd_score, tasa):
    # Devuelve el texto, el color, el fondo suave y lo que se sugiere hacer
    if pd_score < UMBRAL_BAJO:
        return "Riesgo bajo", VERDE, "#E6F4EC", f"Perfil sólido. Se puede aprobar desde {tasa:.1f} % anual."
    if pd_score <= UMBRAL_ALTO:
        return "Riesgo medio", AMBAR, "#FBF3E0", f"Conviene revisarlo con un analista. Si se aprueba, desde {tasa:.1f} % anual."
    return "Riesgo alto", ROJO, "#F9E8E6", f"Considerar rechazo o pedir garantías. Si se aprueba, no bajo {tasa:.1f} % anual."


# ── Estilos ─────────
def aplica_estilos():
    st.markdown(
        """
        <style>
        footer, #MainMenu { visibility: hidden; }
        .block-container { padding-top: 2rem; max-width: 1250px; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF; border-radius: 14px;
            box-shadow: 0 1px 4px rgba(18, 48, 74, 0.10);
        }
        .seccion { color: #12304A; font-size: 1.25rem; font-weight: 700; margin: 6px 0 10px 0; }
        .contexto { color: #5B6670; font-size: 0.95rem; margin-bottom: 12px; }
        .banner { border-radius: 14px; padding: 24px 30px; display: flex; flex-wrap: wrap;
                  justify-content: space-between; align-items: center; gap: 20px; }
        .banner-nivel { font-size: 2rem; font-weight: 800; }
        .banner-frase { color: #1B2733; font-size: 1.05rem; margin-top: 4px; }
        .banner > div:first-child { flex: 2 1 320px; }
        .banner-cifra { text-align: right; flex: 1 1 190px; }
        .leyenda { display: flex; align-items: center; gap: 10px; padding: 4px 0; color: #1B2733; }
        .leyenda:last-child { margin-bottom: 10px; }
        .punto { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
        .etiqueta { color: #5B6670; font-size: 0.9rem; }
        .cifra { color: #12304A; font-size: 2.6rem; font-weight: 800; line-height: 1.1; }
        .cifra-media { color: #12304A; font-size: 1.7rem; font-weight: 700; }
        .pista { color: #5B6670; font-size: 0.9rem; }
        .barra { background: #E6EBEE; border-radius: 8px; height: 12px; margin: 8px 0; }
        .relleno { border-radius: 8px; height: 12px; }
        .pasos { display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; }
        .paso { background: #FFFFFF; border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 180px;
                box-shadow: 0 1px 4px rgba(18, 48, 74, 0.10); }
        .operador { align-self: center; color: #137C86; font-weight: 700; text-align: center; min-width: 70px; }
        .insignia { display: inline-block; background: #12304A; color: #FFFFFF; border-radius: 20px;
                    padding: 3px 14px; font-size: 0.8rem; margin-left: 8px; vertical-align: middle; }
        .ficha { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #EEF1F3; }
        .ficha:last-child { border-bottom: none; }
        .ficha-dato { color: #1B2733; font-weight: 600; }
        .rating { display: inline-block; background: #12304A; color: #FFFFFF; border-radius: 8px;
                  padding: 1px 12px; font-weight: 700; }
        .vacio { background: #FFFFFF; border: 2px dashed #C9D3DA; border-radius: 14px; padding: 36px;
                 text-align: center; color: #5B6670; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Pantalla: cabecera ─────────
def dibuja_cabecera():
    col_logo, col_texto = st.columns([1, 1])
    with col_logo:
        logo = CARPETA / "assets" / "logo_bancandrex.png"
        if logo.exists():  # Si falta el logo, la app sigue funcionando
            st.image(str(logo), width=300)
        else:
            st.header("Bancandrex")
    with col_texto:
        st.markdown(
            "<div class='pista' style='text-align:right; padding-top:26px;'>"
            "Evalúa un préstamo para un cliente del banco y conoce su riesgo en segundos</div>",
            unsafe_allow_html=True,
        )


# ── Pantalla: formulario ─────────
def etiqueta_cliente(id_cliente):
    return "Cliente " + str(id_cliente)


def texto_antiguedad(texto):
    # Los valores vienen en inglés ("10+ years"); aquí se muestran en español
    if texto == "< 1 year":
        return "Menos de 1 año"
    if texto == "10+ years":
        return "10 años o más"
    if texto == "1 year":
        return "1 año"
    if texto.endswith(" years"):
        return texto.replace(" years", " años")
    return "Sin dato"


VERIFICACION = {"Verified": "Verificados", "Source Verified": "Fuente verificada", "Not Verified": "No verificados"}


def fila_ficha(titulo, valor):
    return f"<div class='ficha'><span class='etiqueta'>{titulo}</span><span class='ficha-dato'>{valor}</span></div>"


def dibuja_ficha(ficha):
    if ficha is None:
        st.caption("No se pudo cargar la ficha del cliente.")
        return
    filas = (
        fila_ficha("Calificación (rating)", f"<span class='rating'>{ficha['rating']}</span>")
        + fila_ficha("Ingresos anuales", dinero(ficha["ingresos"]))
        + fila_ficha("Verificación de ingresos", VERIFICACION.get(ficha["ingresos_verificados"], "Sin dato"))
        + fila_ficha("Deuda sobre ingresos (DTI)", f"{ficha['dti']:.1f} %")
        + fila_ficha("Antigüedad en el empleo", texto_antiguedad(ficha["antiguedad_empleo"]))
        + fila_ficha("Líneas de crédito", str(ficha["num_lineas_credito"]))
        + fila_ficha("Uso del crédito rotativo", f"{ficha['porc_uso_revolving']:.1f} %")
        + fila_ficha("Reportes negativos", str(ficha["num_derogatorios"]))
    )
    st.markdown(filas, unsafe_allow_html=True)


def dibuja_formulario(ids):
    col_cliente, col_solicitud = st.columns([4, 6], gap="large")

    with col_cliente:
        with st.container(border=True):
            st.markdown("<div class='seccion'>Cliente del banco</div>", unsafe_allow_html=True)
            id_cliente = st.selectbox("Cliente", ids, format_func=etiqueta_cliente)
            st.caption("Se evalúan clientes que ya tienen historial en el banco (datos ficticios de demostración).")
            dibuja_ficha(carga_ficha(int(id_cliente)))

    with col_solicitud:
        with st.container(border=True):
            st.markdown("<div class='seccion'>Solicitud de préstamo</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="large")
            with col1:
                principal = st.slider("Monto del préstamo (USD)", 500, 40000, 10000, step=500, format="$%d")
                finalidad = st.selectbox("Para qué es el préstamo", list(FINALIDADES.keys()))
            with col2:
                plazo = st.radio("Plazo", list(PLAZOS.keys()), horizontal=True)
                vivienda = st.selectbox("Situación de vivienda", list(VIVIENDAS.keys()))
            st.caption("La tasa no se escribe: el banco la fija según el riesgo, y la app te la sugiere en el resultado.")
            boton = st.button("Evaluar solicitud", type="primary")

        with st.container(border=True):
            st.markdown("<div class='seccion'>Cómo leer el semáforo</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class='leyenda'><span class='punto' style='background:{VERDE};'></span><b>Riesgo bajo:</b> probabilidad de impago menor a {UMBRAL_BAJO * 100:.0f} %</div>
                <div class='leyenda'><span class='punto' style='background:{AMBAR};'></span><b>Riesgo medio:</b> de {UMBRAL_BAJO * 100:.0f} % a {UMBRAL_ALTO * 100:.0f} %</div>
                <div class='leyenda'><span class='punto' style='background:{ROJO};'></span><b>Riesgo alto:</b> mayor a {UMBRAL_ALTO * 100:.0f} %</div>
                """,
                unsafe_allow_html=True,
            )

    solicitud = {
        "id_cliente": int(id_cliente),
        "principal": float(principal),
        "num_cuotas": PLAZOS[plazo],
        "finalidad": FINALIDADES[finalidad],
        "vivienda": VIVIENDAS[vivienda],
    }
    return solicitud, boton


def resumen_solicitud(s, plazo_fijo=None):
    # Una línea con lo que se evaluó, para que nadie confunda el resultado con otro cliente
    plazo = plazo_fijo or str(s["num_cuotas"]) + " meses"
    return f"Cliente {s['id_cliente']} · {dinero(s['principal'])} · {plazo}"


# ── Pantalla: veredicto ─────────
def dibuja_banner(monto, r):
    etiqueta, color, fondo, frase = nivel_riesgo(r["pd_score"], r["tasa_sugerida"])
    por_cada_100 = r["perdida_esperada"] / monto * 100
    st.markdown(
        f"""
        <div class='banner' style='background:{fondo}; border-left: 12px solid {color};'>
            <div>
                <div class='banner-nivel' style='color:{color};'>{etiqueta}</div>
                <div class='banner-frase'>{frase}</div>
            </div>
            <div class='banner-cifra'>
                <div class='etiqueta'>Pérdida esperada</div>
                <div class='cifra'>{dinero(r["perdida_esperada"])}</div>
                <div class='pista'>{porcentaje(r["perdida_esperada"] / monto)} del monto: ${por_cada_100:.2f} por cada $100 prestados</div>
            </div>
            <div class='banner-cifra'>
                <div class='etiqueta'>Tasa sugerida</div>
                <div class='cifra'>{r["tasa_sugerida"]:.1f} %</div>
                <div class='pista'>anual, mínima para cubrir la pérdida esperada</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dibuja_medidor_pd(pd_score):
    # El eje llega a 50 % para que se vean bien las tres zonas del semáforo
    figura = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_score * 100,
            number={"suffix": " %", "valueformat": ".1f", "font": {"color": AZUL}},
            gauge={
                "axis": {
                    "range": [0, 50],
                    "tickvals": [0, 8, 15, 30, 50],
                    "ticktext": ["0 %", "8 %", "15 %", "30 %", "50 %"],
                },
                "bar": {"color": AZUL, "thickness": 0.28},
                "steps": [
                    {"range": [0, UMBRAL_BAJO * 100], "color": VERDE},
                    {"range": [UMBRAL_BAJO * 100, UMBRAL_ALTO * 100], "color": AMBAR},
                    {"range": [UMBRAL_ALTO * 100, 50], "color": ROJO},
                ],
            },
        )
    )
    figura.update_layout(height=190, margin=dict(l=30, r=30, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figura, width="stretch")


def dibuja_medida(titulo, valor, color, explicacion, cifra, cifra_texto):
    st.markdown(
        f"""
        <div class='etiqueta'>{titulo}</div>
        <div class='cifra-media'>{porcentaje(valor)}</div>
        <div class='barra'><div class='relleno' style='width:{valor * 100:.1f}%; background:{color};'></div></div>
        <div class='pista'>{explicacion}</div>
        <div class='cifra-media' style='margin-top:14px; color:{color};'>{cifra}</div>
        <div class='pista'>{cifra_texto}</div>
        """,
        unsafe_allow_html=True,
    )


def dibuja_medidas(monto, r):
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        with st.container(border=True, height=300):
            st.markdown("<div class='etiqueta'>Probabilidad de impago (PD)</div>", unsafe_allow_html=True)
            dibuja_medidor_pd(r["pd_score"])
            st.markdown(
                f"<div class='pista'>De cada 100 préstamos como este, unos {r['pd_score'] * 100:.0f} no se pagarían.</div>",
                unsafe_allow_html=True,
            )
    with col2:
        with st.container(border=True, height=300):
            dibuja_medida(
                "Cuánto debería al fallar (EAD)",
                r["ead_score"],
                AZUL,
                f"Si deja de pagar, aún debería {dinero(monto * r['ead_score'])} de los {dinero(monto)} prestados.",
                dinero(monto * r["ead_score"]),
                "quedarían por pagar al momento de fallar",
            )
    with col3:
        with st.container(border=True, height=300):
            dibuja_medida(
                "Cuánto no se recupera (LGD)",
                r["lgd_score"],
                TURQUESA,
                f"Del saldo pendiente, el banco no lograría recuperar el {porcentaje(r['lgd_score'])}.",
                dinero(monto * r["ead_score"] * r["lgd_score"]),
                "se perderían si el cliente no paga",
            )


def dibuja_cascada(monto, r):
    # Del monto prestado a la pérdida esperada, factor por factor
    exposicion = monto * r["ead_score"]
    perdida_si_falla = exposicion * r["lgd_score"]
    st.markdown("<div class='seccion'>Cómo se llega a la pérdida esperada</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='pasos'>
            <div class='paso'><div class='etiqueta'>Monto prestado</div><div class='cifra-media'>{dinero(monto)}</div></div>
            <div class='operador'>× {porcentaje(r["ead_score"])}<br>EAD</div>
            <div class='paso'><div class='etiqueta'>Exposición al fallar</div><div class='cifra-media'>{dinero(exposicion)}</div></div>
            <div class='operador'>× {porcentaje(r["lgd_score"])}<br>LGD</div>
            <div class='paso'><div class='etiqueta'>Pérdida si no paga</div><div class='cifra-media'>{dinero(perdida_si_falla)}</div></div>
            <div class='operador'>× {porcentaje(r["pd_score"])}<br>PD</div>
            <div class='paso' style='border-top: 5px solid {AMBAR};'><div class='etiqueta'>Pérdida esperada</div><div class='cifra-media'>{dinero(r["perdida_esperada"])}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dibuja_veredicto(solicitud):
    if "evaluacion" not in st.session_state:
        st.markdown(
            "<div class='vacio'><b>Aún no hay resultados</b><br>"
            "Completa los datos de la solicitud y pulsa Evaluar solicitud.</div>",
            unsafe_allow_html=True,
        )
        return

    evaluada, datos, error = st.session_state["evaluacion"]
    if error:
        st.error(error)
        return

    r = datos[0]  # La API responde una lista: aquí enviamos un solo registro
    monto = evaluada["principal"]

    st.markdown("<div class='contexto'>Resultado para: " + resumen_solicitud(evaluada) + "</div>", unsafe_allow_html=True)
    if evaluada != solicitud:
        st.warning("Cambiaste algún dato del formulario. Pulsa Evaluar solicitud para actualizar el resultado.")

    dibuja_banner(monto, r)
    st.write("")
    dibuja_medidas(monto, r)
    st.write("")
    dibuja_cascada(monto, r)


# ── Pantalla: comparar 36 contra 60 meses ─────────
def dibuja_tarjeta_plazo(plazo, monto, r, es_menor):
    etiqueta, color, fondo, frase = nivel_riesgo(r["pd_score"], r["tasa_sugerida"])
    insignia = "<span class='insignia'>Menor pérdida esperada</span>" if es_menor else ""
    st.markdown(
        f"""
        <div class='paso' style='border-top: 8px solid {color}; padding: 22px 26px;'>
            <div class='cifra-media'>{plazo} meses{insignia}</div>
            <div style='color:{color}; font-weight:700; font-size:1.2rem; margin-bottom:14px;'>{etiqueta}</div>
            <div class='etiqueta'>Probabilidad de impago</div>
            <div class='cifra-media'>{porcentaje(r["pd_score"])}</div>
            <div class='etiqueta' style='margin-top:14px;'>Pérdida esperada</div>
            <div class='cifra'>{dinero(r["perdida_esperada"])}</div>
            <div class='pista'>{porcentaje(r["perdida_esperada"] / monto)} del monto prestado</div>
            <div class='etiqueta' style='margin-top:14px;'>Tasa sugerida</div>
            <div class='cifra-media'>{r["tasa_sugerida"]:.1f} % anual</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dibuja_comparador(solicitud):
    st.markdown(
        "<div class='contexto'>Con los mismos datos del formulario, mira qué cambia si el plazo es de 36 o de 60 meses.</div>",
        unsafe_allow_html=True,
    )
    if st.button("Comparar 36 y 60 meses"):
        registros = []
        for plazo in [36, 60]:
            registros.append(arma_registro(solicitud, plazo))
        datos, error = calcula(registros)
        st.session_state["comparacion"] = (solicitud, datos, error)

    if "comparacion" not in st.session_state:
        st.markdown(
            "<div class='vacio'><b>Aún no hay comparación</b><br>Pulsa Comparar 36 y 60 meses.</div>",
            unsafe_allow_html=True,
        )
        return

    comparada, datos, error = st.session_state["comparacion"]
    if error:
        st.error(error)
        return

    # La API responde en el mismo orden en que enviamos: primero 36 y luego 60
    monto = comparada["principal"]
    pe_36 = datos[0]["perdida_esperada"]
    pe_60 = datos[1]["perdida_esperada"]

    st.markdown(
        "<div class='contexto'>Comparación para: " + resumen_solicitud(comparada, "36 y 60 meses") + "</div>",
        unsafe_allow_html=True,
    )
    if comparada != solicitud:
        st.warning("Cambiaste algún dato del formulario. Pulsa Comparar 36 y 60 meses para actualizar.")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        dibuja_tarjeta_plazo(36, monto, datos[0], pe_36 <= pe_60)
    with col2:
        dibuja_tarjeta_plazo(60, monto, datos[1], pe_60 < pe_36)

    diferencia = abs(pe_60 - pe_36)
    plazo_mayor = "60" if pe_60 > pe_36 else "36"
    st.markdown(
        f"<div class='contexto' style='margin-top:14px;'>Con {plazo_mayor} meses la pérdida esperada es {dinero(diferencia)} mayor.</div>",
        unsafe_allow_html=True,
    )


# ── Programa principal ─────────
favicon = CARPETA / "assets" / "favicon_bancandrex.png"
if favicon.exists():
    st.set_page_config(page_title="Bancandrex", page_icon=str(favicon), layout="wide")
else:
    st.set_page_config(page_title="Bancandrex", layout="wide")
aplica_estilos()
dibuja_cabecera()
ids = carga_clientes()

solicitud, evaluar = dibuja_formulario(ids)
if evaluar:
    datos, error = calcula([arma_registro(solicitud, solicitud["num_cuotas"])])
    st.session_state["evaluacion"] = (solicitud, datos, error)

st.write("")
pestana_veredicto, pestana_comparar = st.tabs(["Evaluar solicitud", "Comparar plazos"])
with pestana_veredicto:
    dibuja_veredicto(solicitud)
with pestana_comparar:
    dibuja_comparador(solicitud)

st.write("")
with st.expander("¿Cómo se calcula la pérdida esperada?"):
    st.write("Pérdida esperada = probabilidad de impago (PD) × monto × exposición al fallar (EAD) × pérdida si no paga (LGD).")
    st.write("Los cortes del semáforo sobre la PD son: bajo menos de 8 %, medio de 8 % a 15 %, alto más de 15 %.")
with st.expander("¿Cómo se calcula la tasa sugerida?"):
    st.write("Tasa sugerida = costo del dinero + costos de operar + ganancia + pérdida esperada por año.")
    st.write("La pérdida esperada se pasa a porcentaje del monto y se divide entre los años del plazo. El costo del dinero, los costos y la ganancia son supuestos ilustrativos fijados en la API: no son la política de ningún banco real.")
    st.write("El modelo no usa la tasa para predecir el riesgo: la tasa la fija el banco al conceder, a partir del riesgo, y usarla como dato sería circular.")
with st.expander("Para el equipo técnico"):
    st.write("API en uso: " + API_BASE_URL)
    st.write("En Render, la variable de entorno API_BASE_URL debe tener la URL pública de la API.")
