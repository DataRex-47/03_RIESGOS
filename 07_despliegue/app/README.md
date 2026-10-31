# Bancandrex, Mesa de decisión de crédito (app Streamlit)

App para que un analista elija un **cliente que ya existe en el banco**, vea su ficha, simule un préstamo y obtenga el nivel de riesgo, la probabilidad de impago, la pérdida esperada, la **tasa sugerida** y una comparación entre 36 y 60 meses. La app no tiene lógica de Machine Learning: pide todo a la API.

## Qué se ve en la pantalla

| Zona | Contenido |
|---|---|
| Cliente del banco | Selector de cliente y su ficha: rating, ingresos, verificación, DTI, antigüedad, líneas de crédito, uso rotativo y reportes negativos. Son datos ficticios de demostración |
| Solicitud de préstamo | Monto (500 a 40.000), plazo (36 o 60 meses), finalidad y situación de vivienda |
| Resultado | Semáforo (bajo, medio, alto), pérdida esperada, tasa sugerida, medidor de la PD, barras de EAD y LGD y cascada de la pérdida esperada |
| Comparar plazos | Las mismas cifras a 36 y a 60 meses, lado a lado |

**La tasa no se escribe.** La fija el banco según el riesgo del cliente, así que es un resultado: "se puede aprobar desde X % anual". Es una tasa mínima de ejemplo (fondeo 3,0 %, costos 0,7 % y margen 1,3 % son supuestos ilustrativos más la pérdida esperada por año), no una política de un banco real.

## Ejecutar en local

Hacen falta dos terminales abiertas a la vez, con el entorno activado.

| Terminal | Carpeta | Comando |
|---|---|---|
| 1, la API | `07_despliegue\API` | `uv run uvicorn main:app --reload` |
| 2, la app | `07_despliegue\app` | `uv run streamlit run app.py` |

Antes de arrancar la app, comprueba que la API responde: abre http://127.0.0.1:8000/health (debe mostrar `{"status":"OK"}`) y http://127.0.0.1:8000/clientes (debe mostrar los ids).

Si faltan paquetes, desde la raíz del proyecto:

```
uv add streamlit plotly requests fastapi uvicorn pydantic cloudpickle
```

Si `uvicorn` dice `Attribute "app" not found in module "main"`, estás en la carpeta equivocada: en la raíz del proyecto hay otro `main.py`. Entra primero a `07_despliegue\API`.

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | Toda la app: configuración, cliente HTTP, formulario y pantallas |
| `assets/` | Logo y favicon de Bancandrex |
| `.streamlit/config.toml` | Colores de la app y modo sin barra de herramientas |
| `idea/design_spec.json` | El diseño acordado |
| `requirements.txt` | Paquetes con versión exacta para Render |

## Cambiar los cortes del semáforo

Están al inicio de `app.py`: `UMBRAL_BAJO = 0.08` y `UMBRAL_ALTO = 0.15`, sobre la probabilidad de impago. Con los modelos del 31/10/2026 dejan 38 % de los préstamos en riesgo bajo, 37 % en medio y 25 % en alto.

## Desplegar en Render

El código no cambia entre local y Render: la URL de la API se lee de la variable de entorno `API_BASE_URL`. Si no existe, la app usa la API local (`http://127.0.0.1:8000`).

1. Haz push de `07_despliegue/API/` (código y los tres `.pkl` nuevos). La API de Render necesita las rutas `/clientes` y `/clientes/{id}` y el campo `tasa_sugerida`. Espera a que termine su redeploy y comprueba https://zero3-riesgos.onrender.com/clientes/4155468.
2. Haz push de `07_despliegue/app/`.
3. En Render, el servicio de la app ya existe con estos datos (si hay que crearlo de cero, son los mismos):

| Campo | Valor |
|---|---|
| Root Directory | `07_despliegue/app` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |

4. En Environment:

| Clave | Valor |
|---|---|
| `PYTHON_VERSION` | `3.12.13` (el mismo de la API) |
| `API_BASE_URL` | `https://zero3-riesgos.onrender.com` (sin barra al final) |

## Servicio dormido (cold start)

En el plan gratuito Render duerme el servicio. La primera consulta puede tardar hasta un minuto: la app despierta la API sola antes de cada cálculo y muestra un mensaje de espera. La app también se duerme: la primera visita tras un rato sin uso tarda en cargar.

## Límites que conviene conocer

- Los clientes son ficticios y ya existen en el historial de demostración. La app no registra clientes nuevos.
- Los plazos son solo 36 y 60 meses porque los datos de entrenamiento no tienen otros.
- El EAD y el LGD son casi constantes (66 % y 89 %): el que ordena el riesgo es la PD, con un AUC cercano a 0,70.
