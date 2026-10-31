# Informe de despliegue de la API

## 1. El flujo completo

| Paso | Archivo | Qué hace | Qué produce |
|---|---|---|---|
| 1 | `API/main.py` y `API/schemas.py` | Recibe una lista JSON y valida los seis campos enviados por la app. | Solicitudes válidas o error HTTP 422. |
| 2 | `API/main.py` y `API/demo_clientes.csv` | Busca el historial asociado a `id_cliente`; si no existe, responde 404. | Ocho variables históricas para completar la solicitud. |
| 3 | `API/main.py` | Calcula `imp_cuota` con la fórmula de cuota mensual del sistema francés y prepara las columnas del modelo. | DataFrame con identificador y variables de scoring. |
| 4 | `API/scoring.py` y los tres artefactos `.pkl` | Carga PD, EAD y LGD, limita ingresos a 400000 y calcula los tres scores y la pérdida esperada. | DataFrame con cinco columnas de salida. |
| 5 | `API/main.py` | Convierte la salida a JSON y responde al endpoint. | Lista JSON de objetos en `POST /predict`. |
| 6 | Render | Sirve la misma aplicación publicada. | API disponible en https://zero3-riesgos.onrender.com/ . |

## 2. Cómo leer cada archivo

| Archivo | Qué es, en cristiano | Para qué sirve |
|---|---|---|
| `__init__.py` | Marca la carpeta como paquete Python. | Permite importar módulos de la API como paquete. |
| `scoring.py` | Adaptación del motor de `02_produccion_scoring.py`, colocada junto a la API porque Render despliega esa carpeta. | Carga una vez los tres modelos y devuelve un DataFrame desde `scoring_df(datos)`. |
| `schemas.py` | Define los campos de solicitud, tipos, categorías y límites admitidos. | Rechaza campos desconocidos y valores fuera del contrato antes de puntuar. |
| `main.py` | La ventanilla de la API. Valida, busca historial, completa la entrada, calcula la cuota y llama al motor. | Expone `/health` y `POST /predict`. |
| `requirements.txt` | Lista versiones fijadas de las dependencias de la API, el scoring y `requests==2.34.2`. | Instala el entorno del servicio y permite ejecutar ambos clientes de prueba. |
| `test_payload.json` | Solicitud de ejemplo en formato de lista con un registro. | Sirve para probar `/predict`; contiene un identificador de la tabla de demostración. No copiar ese identificador a documentación pública. |
| `demo_clientes.csv` | Historial de demostración, no una base de datos productiva. | Permite resolver el historial por `id_cliente` durante la prueba. |
| `cliente_local.py` | Cliente HTTP ligero que lee `test_payload.json` y llama a `127.0.0.1:8000`. | Prueba el servicio local. |
| `cliente_render.py` | Cliente HTTP que lee el mismo payload y llama a `https://zero3-riesgos.onrender.com/predict`. | Prueba el servicio publicado en Render. |
| `artefacto_pipeline_pd.pkl` | Pipeline serializado de probabilidad de impago. | Produce `pd_score`. |
| `artefacto_pipeline_ead.pkl` | Pipeline serializado de exposición al impago. | Produce `ead_score`. |
| `artefacto_pipeline_lgd.pkl` | Pipeline serializado de pérdida dado el impago. | Produce `lgd_score`. |

## 3. El contrato

La API recibe siempre una lista JSON, incluso para un único registro. Los ejemplos de esta tabla son sintéticos; el identificador debe existir en el historial cargado por el servicio.

| Campo de entrada | De dónde llega | Tipo y aceptación | Ejemplo sintético |
|---|---|---|---|
| `id_cliente` | App | Entero; debe encontrarse en el historial. | `123456` |
| `principal` | App | Número entre 500 y 40000. | `12000.0` |
| `tipo_interes` | App | Número entre 5 y 31, porcentaje anual. | `12.5` |
| `num_cuotas` | App | Entero, solo 36 o 60; internamente se convierte al texto esperado por el modelo. | `36` |
| `finalidad` | App | Texto: `debt_consolidation`, `credit_card`, `home_improvement`, `major_purchase`, `small_business`, `car`, `medical`, `moving`, `vacation`, `house`, `other`, `wedding`, `educational` o `renewable_energy`. | `debt_consolidation` |
| `vivienda` | App | Texto: `RENT`, `OWN`, `MORTGAGE`, `ANY`, `NONE` u `OTHER`. | `RENT` |
| `ingresos` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `ingresos_verificados` | Historial | Texto del registro asociado. | Valor del historial |
| `rating` | Historial | Texto del registro asociado. | Valor del historial |
| `dti` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `antigüedad_empleo` | Historial | Texto del registro asociado; conserva la ü requerida por el artefacto. | Valor del historial |
| `num_lineas_credito` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `porc_uso_revolving` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `num_derogatorios` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `imp_cuota` | Calculado | Número; se calcula a partir de principal, interés y plazo. | Cuota calculada |

| Campo de respuesta | Tipo | Significado |
|---|---|---|
| `id_cliente` | Entero | Identificador enviado por la app. |
| `pd_score` | Número entre 0 y 1 | Probabilidad estimada de impago. |
| `ead_score` | Número entre 0 y 1 | Proporción estimada de exposición. |
| `lgd_score` | Número entre 0 y 1 | Proporción estimada de pérdida dado el impago. |
| `perdida_esperada` | Número | `pd_score × principal × ead_score × lgd_score`. |

La respuesta observada en una prueba de scoring tuvo estas columnas. El identificador se sustituyó por `0` para no publicar el identificador de la fila de demostración; los cuatro resultados numéricos sí son los devueltos por esa prueba:

```json
[
  {
    "id_cliente": 0,
    "pd_score": 0.03813762855132282,
    "ead_score": 0.5548808941357176,
    "lgd_score": 0.8804758680802829,
    "perdida_esperada": 279.4873605590689
  }
]
```

## 4. De dónde sale el historial

`demo_clientes.csv` contiene datos ficticios y simula la consulta de historial por identificador; es adecuado para probar la demo y no representa clientes reales.
Si la API se adapta para puntuar clientes reales, el historial debe consultarse en una base de datos protegida; no debe usarse el CSV de entrenamiento.

## 5. Qué se validó y qué no

| Prueba | Resultado |
|---|---|
| Importar `main` localmente y cargar los artefactos | Correcto. |
| `GET /health` local | HTTP 200, `status: OK`. |
| `POST /predict` local con `test_payload.json` | HTTP 200, una fila y las cinco columnas documentadas. |
| `GET /health` en Render | HTTP 200, `status: OK`. |
| `GET /docs` y `GET /openapi.json` en Render | Ambos HTTP 200. |
| `POST /predict` en Render con `test_payload.json` | HTTP 200, una fila y las cinco columnas documentadas. |
| Variable `PYTHON_VERSION` de Render | El usuario confirmó que configuró `3.12.13`; no se inspeccionó directamente el runtime. |
| `cliente_local.py` | Ejecutado con `uv run`; recibió HTTP 200 y la salida esperada. |
| `cliente_render.py` | Ejecutado con `uv run`; recibió HTTP 200 y la salida esperada desde Render. |
| Endpoint `/debug` | No existe en `main.py`. |
| Respuesta 422 para datos inválidos | No probada. |

## 6. Cómo probarla

Desde la raíz del repositorio, iniciar el servicio local en una terminal:

```powershell
Push-Location '07_despliegue\API'
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

En otra terminal, comprobar salud y enviar el payload de ejemplo:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
$body = Get-Content '07_despliegue\API\test_payload.json' -Raw
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $body
```

Debe verse `status: OK` y una lista con `id_cliente`, `pd_score`, `ead_score`, `lgd_score` y `perdida_esperada`. Estas comprobaciones se ejecutaron también contra `https://zero3-riesgos.onrender.com/health` y `https://zero3-riesgos.onrender.com/predict`.

Con el servicio local corriendo, los dos clientes se ejecutan desde la raíz del repositorio:

```powershell
uv run python '07_despliegue\API\cliente_local.py'
uv run python '07_despliegue\API\cliente_render.py'
```

Ambos deben mostrar `Código HTTP: 200` y una respuesta JSON con las cinco columnas de salida.

Configuración de Render indicada para el servicio. La URL respondió correctamente; los demás ajustes no se inspeccionaron directamente en el dashboard:

| Ajuste | Valor |
|---|---|
| Tipo | Web Service |
| Root Directory | `07_despliegue/API` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| `PYTHON_VERSION` | `3.12.13`, configurado por el usuario en Environment de Render |
| Health Check Path | `/health` |

## 7. Pendientes

1. El historial ficticio es adecuado para la demo; si se habilita scoring de clientes reales, conectarlo a una base de datos protegida.
2. La API entrega scores sin semáforo; la lógica de umbrales corresponde a la app de Streamlit.
