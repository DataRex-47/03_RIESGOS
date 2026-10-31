# Bancandrex, Mesa de decisión de crédito (app Streamlit)

App para que un analista elija un cliente, simule un préstamo y vea el nivel de riesgo, la probabilidad de impago, la pérdida esperada y una comparación entre 36 y 60 meses. La app no tiene lógica de Machine Learning: pide todo a la API.

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

Están al inicio de `app.py`: `UMBRAL_BAJO = 0.08` y `UMBRAL_ALTO = 0.15`, sobre la probabilidad de impago.

## Desplegar en Render

El código no cambia entre local y Render: la URL de la API se lee de una variable de entorno. Si no existe, la app usa la API local.

1. Haz push de `07_despliegue/API/main.py`. La API de Render necesita la ruta `/clientes`. Espera a que termine su redeploy y comprueba https://zero3-riesgos.onrender.com/clientes.
2. Haz push de `07_despliegue/app/`.
3. En Render crea un Web Service nuevo con el mismo repositorio y estos datos:

| Campo | Valor |
|---|---|
| Root Directory | `07_despliegue/app` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |

4. En Environment agrega dos variables:

| Clave | Valor |
|---|---|
| `PYTHON_VERSION` | `3.12.13` (el mismo de la API) |
| `API_BASE_URL` | `https://zero3-riesgos.onrender.com` (sin barra al final) |

## Servicio dormido (cold start)

En el plan gratuito Render duerme el servicio. La primera consulta puede tardar hasta un minuto: la app despierta la API sola antes de cada cálculo y muestra un mensaje de espera. La app también se duerme: la primera visita tras un rato sin uso tarda en cargar.
