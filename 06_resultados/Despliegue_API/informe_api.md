# Informe de despliegue de la API

Actualizado el 31/10/2026: la tasa de interés ya no es una entrada, es una respuesta (`tasa_sugerida`).

## 1. El flujo completo

| Paso | Archivo | Qué hace | Qué produce |
|---|---|---|---|
| 1 | `API/main.py` y `API/schemas.py` | Recibe una lista JSON y valida los cinco campos enviados por la app. Rechaza cualquier campo extra, por ejemplo `tipo_interes`. | Solicitudes válidas o error HTTP 422. |
| 2 | `API/main.py` y `API/demo_clientes.csv` | Busca el historial asociado a `id_cliente`; si no existe, responde 404. | Ocho variables históricas para completar la solicitud. |
| 3 | `API/main.py` | Une solicitud e historial y convierte el plazo al texto que espera el modelo. | DataFrame con identificador y variables de scoring. |
| 4 | `API/scoring.py` y los tres artefactos `.pkl` | Carga PD, EAD y LGD, limita ingresos a 400.000 y calcula los tres scores y la pérdida esperada. | DataFrame con cinco columnas. |
| 5 | `API/scoring.py` (`calcula_tasa_sugerida`) | Convierte la pérdida esperada en una tasa anual mínima (sección 4). | La columna `tasa_sugerida`. |
| 6 | `API/main.py` | Convierte la salida a JSON y responde al endpoint. | Lista JSON de objetos en `POST /predict`. |
| 7 | Render | Sirve la misma aplicación publicada. | API disponible en https://zero3-riesgos.onrender.com/ . |

## 2. Rutas de la API

| Ruta | Para qué sirve | Respuesta |
|---|---|---|
| `GET /health` | Comprobar que el servicio está vivo | `{"status": "OK"}` |
| `GET /clientes` | Llenar el selector de la app | `{"ids": [...]}`, solo identificadores |
| `GET /clientes/{id_cliente}` | Ficha del cliente: lo que el banco ya sabe de él | Ingresos, verificación, rating, DTI, antigüedad, líneas de crédito, uso rotativo y reportes negativos. 404 si no existe |
| `POST /predict` | Puntuar una o varias solicitudes | Lista con los scores, la pérdida esperada y la tasa sugerida |

## 3. Cómo leer cada archivo

| Archivo | Qué es, en cristiano | Para qué sirve |
|---|---|---|
| `__init__.py` | Marca la carpeta como paquete Python. | Permite importar módulos de la API como paquete. |
| `scoring.py` | Adaptación del motor de `02_produccion_scoring.py`, colocada junto a la API porque Render despliega esa carpeta. | Carga una vez los tres modelos, devuelve un DataFrame desde `scoring_df(datos)` y calcula la tasa sugerida. |
| `schemas.py` | Define los campos de solicitud, tipos, categorías y límites admitidos. | Rechaza campos desconocidos y valores fuera del contrato antes de puntuar. |
| `main.py` | La ventanilla de la API. Valida, busca historial, completa la entrada y llama al motor. | Expone `/health`, `/clientes`, `/clientes/{id}` y `POST /predict`. |
| `requirements.txt` | Lista versiones fijadas de las dependencias de la API, el scoring y `requests==2.34.2`. | Instala el entorno del servicio y permite ejecutar ambos clientes de prueba. |
| `test_payload.json` | Solicitud de ejemplo en formato de lista con un registro. | Sirve para probar `/predict`; contiene un identificador de la tabla de demostración. No copiar ese identificador a documentación pública. |
| `demo_clientes.csv` | Historial de demostración, no una base de datos productiva. | Permite resolver el historial por `id_cliente` durante la prueba. |
| `cliente_local.py` | Cliente HTTP ligero que lee `test_payload.json` y llama a `127.0.0.1:8000`. | Prueba el servicio local. |
| `cliente_render.py` | Cliente HTTP que lee el mismo payload y llama a `https://zero3-riesgos.onrender.com/predict`. | Prueba el servicio publicado en Render. |
| `artefacto_pipeline_pd.pkl` | Pipeline serializado de probabilidad de impago. | Produce `pd_score`. |
| `artefacto_pipeline_ead.pkl` | Pipeline serializado de exposición al impago. | Produce `ead_score`. |
| `artefacto_pipeline_lgd.pkl` | Pipeline serializado de pérdida dado el impago. | Produce `lgd_score`. |

## 4. El contrato

La API recibe siempre una lista JSON, incluso para un único registro. Los ejemplos de esta tabla son sintéticos; el identificador debe existir en el historial cargado por el servicio.

**Qué se envía y qué no.** El cliente ya existe en el banco: la app solo manda quién es y qué se quiere prestar. Lo que el banco ya sabe del cliente (ingresos, rating, etcétera) lo pone la API desde el historial. La tasa **no** se envía: la decide el banco según el riesgo, así que es una respuesta.

| Campo de entrada | De dónde llega | Tipo y aceptación | Ejemplo sintético |
|---|---|---|---|
| `id_cliente` | App | Entero; debe encontrarse en el historial. | `123456` |
| `principal` | App | Número entre 500 y 40.000, el rango que vio el modelo al entrenar. | `12000.0` |
| `num_cuotas` | App | Entero, solo 36 o 60 (los únicos plazos que hay en los datos); internamente se convierte al texto esperado por el modelo. | `36` |
| `finalidad` | App | Texto: `debt_consolidation`, `credit_card`, `home_improvement`, `major_purchase`, `small_business`, `car`, `medical`, `moving`, `vacation`, `house`, `other`, `wedding`, `educational` o `renewable_energy`. | `debt_consolidation` |
| `vivienda` | App | Texto: `RENT`, `OWN`, `MORTGAGE`, `ANY`, `NONE` u `OTHER`. Se pide en la solicitud porque el historial de la demo no lo trae. | `RENT` |
| `ingresos` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `ingresos_verificados` | Historial | Texto del registro asociado. | Valor del historial |
| `rating` | Historial | Texto del registro asociado. | Valor del historial |
| `dti` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `antigüedad_empleo` | Historial | Texto del registro asociado; conserva la ü requerida por el artefacto. | Valor del historial |
| `num_lineas_credito` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `porc_uso_revolving` | Historial | Variable numérica del registro asociado. | Valor del historial |
| `num_derogatorios` | Historial | Variable numérica del registro asociado. | Valor del historial |

Ya no existen `tipo_interes` ni `imp_cuota`. Si alguien envía `tipo_interes`, la API responde 422 (`Extra inputs are not permitted`).

| Campo de respuesta | Tipo | Significado |
|---|---|---|
| `id_cliente` | Entero | Identificador enviado por la app. |
| `pd_score` | Número entre 0 y 1 | Probabilidad estimada de impago. |
| `ead_score` | Número entre 0 y 1 | Proporción estimada de exposición. |
| `lgd_score` | Número entre 0 y 1 | Proporción estimada de pérdida dado el impago. |
| `perdida_esperada` | Número | `pd_score × principal × ead_score × lgd_score`. |
| `tasa_sugerida` | Número | Tasa anual mínima, en %, que cubre la pérdida esperada, los costos y la ganancia. |

Respuesta real de la prueba local con el cliente de demostración, 15.000 a 36 meses. El identificador se sustituyó por `0` para no publicar el de la fila de demostración; los cinco resultados numéricos son los devueltos:

```json
[
  {
    "id_cliente": 0,
    "pd_score": 0.038675085270861706,
    "ead_score": 0.5828951424607313,
    "lgd_score": 0.8795211961041927,
    "perdida_esperada": 297.412546396778,
    "tasa_sugerida": 5.660916769770617
  }
]
```

## 5. Cómo se calcula la tasa sugerida

La tasa que el banco cobra se construye con el método de costo más margen:

```
tasa sugerida = costo de fondeo + costos operativos + margen + pérdida esperada anual
```

| Pieza | Valor | De dónde sale |
|---|---|---|
| Costo de fondeo | 3,0 % anual | Supuesto ilustrativo: lo que le cuesta al banco conseguir el dinero |
| Costos operativos | 0,7 % anual | Supuesto ilustrativo: lo que cuesta operar el préstamo |
| Margen | 1,3 % anual | Supuesto ilustrativo: la ganancia del banco |
| Pérdida esperada anual | Depende del cliente | `100 × perdida_esperada / principal / (num_cuotas / 12)`: el riesgo del cliente, repartido por año |

Los tres supuestos son del autor, con fines de demostración, y están como constantes al inicio de `scoring.py` (`COSTO_FONDEO`, `COSTOS_OPERATIVOS`, `MARGEN`). Un banco real los toma de tesorería y de su política comercial.

**Qué es y qué no es.** Es una tasa **mínima** para no perder dinero con el cliente. No incluye la pérdida inesperada ni el capital que el banco debe reservar, ni compara con la competencia. Por eso para clientes de riesgo bajo da valores cercanos al 5,7 %, por debajo de lo que aparece en los datos históricos (de 5 % a 31 %). Para un cliente de riesgo alto sube bastante.

**Por qué importa que sea una respuesta.** En la primera versión la tasa entraba al modelo, que aprendió que a mayor tasa menor riesgo. Eso es circular, porque la tasa se fija a partir del riesgo. Ahora el modelo mide el riesgo con lo que se conoce del cliente y la tasa se deriva de ese riesgo.

Ejemplo con el mismo cliente de demostración y 15.000 de principal:

| Plazo | PD | Pérdida esperada | Tasa sugerida |
|---|---|---|---|
| 36 meses | 3,9 % | 297,41 | 5,66 % |
| 60 meses | 3,9 % | 352,60 | 5,47 % |

A 60 meses la pérdida esperada total es mayor (el EAD sube porque se debe más cuando el cliente falla), pero repartida en más años da una tasa anual algo menor.

## 6. De dónde sale el historial

`demo_clientes.csv` contiene datos ficticios y simula la consulta de historial por identificador; es adecuado para probar la demo y no representa clientes reales. Los clientes de la demo ya existen en el banco: la app no registra clientes nuevos.
Si la API se adapta para puntuar clientes reales, el historial debe consultarse en una base de datos protegida; no debe usarse el CSV de entrenamiento. Subir datos reales de clientes a un repositorio público no es aceptable.

## 7. Qué se validó y qué no

| Prueba | Resultado |
|---|---|
| Cargar `main` y los artefactos nuevos (Python 3.12) | Correcto. |
| `GET /health` local | HTTP 200, `status: OK`. |
| `GET /clientes` y `GET /clientes/{id}` locales | 200; 50 ids; la ficha trae los ocho datos. |
| `GET /clientes/{id}` con un id inexistente | 404. |
| `POST /predict` local con `test_payload.json` | HTTP 200, una fila y las seis columnas documentadas. |
| `POST /predict` con `tipo_interes` | HTTP 422, `Extra inputs are not permitted`. |
| Comparación a 36 y 60 meses | Dos respuestas distintas y coherentes (sección 5). |
| Render con los modelos nuevos | Sin validar: se valida después del push. |
| Variable `PYTHON_VERSION` de Render | El usuario confirmó que configuró `3.12.13`; no se inspeccionó directamente el runtime. |
| Endpoint `/debug` | No existe en `main.py`. |

## 8. Cómo probarla

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

Debe verse `status: OK` y una lista con `id_cliente`, `pd_score`, `ead_score`, `lgd_score`, `perdida_esperada` y `tasa_sugerida`.

Con el servicio local corriendo, los dos clientes se ejecutan desde la raíz del repositorio:

```powershell
uv run python '07_despliegue\API\cliente_local.py'
uv run python '07_despliegue\API\cliente_render.py'
```

Ambos deben mostrar `Código HTTP: 200` y una respuesta JSON con las seis columnas de salida. `cliente_render.py` solo funcionará después de subir los cambios y de que Render termine de redesplegar.

Configuración de Render indicada para el servicio. La URL respondió correctamente antes del cambio; los demás ajustes no se inspeccionaron directamente en el dashboard:

| Ajuste | Valor |
|---|---|
| Tipo | Web Service |
| Root Directory | `07_despliegue/API` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| `PYTHON_VERSION` | `3.12.13`, configurado por el usuario en Environment de Render |
| Health Check Path | `/health` |

## 9. Pendientes

1. El historial ficticio es adecuado para la demo; si se habilita scoring de clientes reales, conectarlo a una base de datos protegida.
2. La API entrega scores y tasa sugerida sin semáforo; la lógica de umbrales corresponde a la app de Streamlit.
3. Los supuestos de la tasa sugerida son ilustrativos; para un uso real hay que reemplazarlos por los de tesorería y añadir la pérdida inesperada.
