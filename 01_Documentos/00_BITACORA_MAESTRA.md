---
tags: [bitacora, scoring-riesgos, caso-ml, contexto]
aliases: [Bitácora Scoring de Riesgos, Contexto del Proyecto 3]
date: 2026-10-11
updated: 2026-10-31
status: 🌱 (Seed)
---

# 🧭 Bitácora Maestra: Caso 3 · Scoring de Riesgos

> 📍 **Bitacorizado hasta:** reentrenamiento sin tasa ni cuota, API con tasa sugerida y app con ficha del cliente, todo documentado (31/10/2026) · siguiente paso: push, validar en Render y confirmar los cortes del semáforo

> [!SUMMARY] Para qué sirve
> Reengancharte en 3 minutos y practicar sin releer los notebooks.
> Cada fase trae: pregunta de recall, qué pasó, qué quedó en disco, cómo se hizo y un ejercicio.
> **El contexto del caso (negocio, fórmula, sectores) no está aquí**: está en el vault, en `Casos_ML/Caso_03_Scoring_Riesgos.md` y en la carpeta `Riesgo_y_Banca/`. Aquí solo va el seguimiento. La guía corta en PDF está en esta misma carpeta: `Scoring de Riesgos Guia.pdf`.

---

## 📍 Dónde estoy

**Fases 01 a 04 cerradas (17/10/2026).** Importación, Calidad, EDA y Transformación hechas. **Fases 05 y 06 omitidas por decisión (24/10/2026).** **Modelización, preproducción, reentrenamiento y scoring cerrados (27/10/2026):** los tres modelos están entrenados y sus artefactos y scripts están en `07_despliegue/`.

**API y app cerradas (31/10/2026):** la API (A_11) y la app del gestor (A_12) están en Render. El semáforo vive en la app, con cortes de 8 % y 15 % sobre la PD.

**Cambio grande (31/10/2026): la tasa sale de los modelos.** `tipo_interes` e `imp_cuota` dejaron de ser predictoras (la tasa la fija el banco según el riesgo, así que usarla para predecir el riesgo era circular). Se reentrenó todo, la API devuelve una `tasa_sugerida` y la app ya no tiene control de tasa. README del proyecto escrito.

**Siguiente acción:** push (carpeta API primero y luego la app), validar en Render, confirmar los cortes del semáforo y el informe corto de arquitectura.

**Contrato de entrada vigente (31/10/2026):** 5 campos por la app (`id_cliente`, `principal` de 500 a 40.000, `num_cuotas` 36 o 60, `finalidad`, `vivienda`) y 8 por el historial del cliente en CSV (`ingresos`, `ingresos_verificados`, `rating`, `dti`, `antigüedad_empleo`, `num_lineas_credito`, `porc_uso_revolving`, `num_derogatorios`). Salida: `pd_score`, `ead_score`, `lgd_score`, `perdida_esperada` y `tasa_sugerida`. Detalle en `07_despliegue/variables_entrada_backend.md`.

> [!IMPORTANT] 📌 Para releer: la idea que lo ordena todo
> **Valor esperado = probabilidad de que ocurra un evento × impacto económico si ocurre.**
> En banca se parte el impacto en dos: **Pérdida esperada = PD × Principal × EAD × LGD**, con tres modelos (una clasificación y dos regresiones). En otros sectores son dos modelos. Detalle: `Riesgo_y_Banca/Valor_Esperado_Cualquier_Negocio.md` (vault).

---

## 🔓 Decisiones abiertas

| Decisión | Opciones | Quién decide | Qué bloquea | Desde |
|---|---|---|---|---|
| Cortes del semáforo (propuesta puesta en la app: 8 % y 15 %) | Confirmarlos o moverlos. Con los modelos nuevos y los 200.000 préstamos: 38,0 % bajo, 37,2 % medio y 24,8 % alto | Rex | Nada técnico: son dos constantes al inicio de `app.py` | 27/10 |
| Supuestos de la tasa sugerida (fondeo 3,0 %, costos 0,7 %, margen 1,3 %) | Dejarlos como ilustrativos o ajustarlos | Rex | Nada técnico: son tres constantes en `API/scoring.py` | 31/10 |

### ✅ Decisiones ya cerradas

| Decisión | Resultado |
|---|---|
| Dónde vive el semáforo | En la app. La API devuelve solo PD, EAD, LGD y pérdida esperada |
| Escritura de `antigüedad_empleo` | Con ü en todo el contrato, como los artefactos |
| Cómo llegan artefactos e historial a Render | Los 3 `.pkl` y `demo_clientes.csv` (50 clientes ficticios) van dentro de `07_despliegue/API/` |
| Lista de clientes de la app | Ruta `GET /clientes` de la API, que devuelve solo los ids |
| Qué hacer con la tasa (31/10) | Sale de los modelos y de la entrada. La API la devuelve como `tasa_sugerida` = fondeo + costos + margen + pérdida esperada anual. Los modelos se reentrenaron sin `tipo_interes` ni `imp_cuota` |
| Ficha del cliente (31/10) | Ruta `GET /clientes/{id}` y tarjeta en la app. El cliente ya existe en el banco |
| Formato de `num_cuotas` | La API lo pasa al modelo como categoría (" 36 months") |


---

## 🔗 La cadena

```
A_01 Import > A_02 Calidad > A_03 EDA > A_04 Transform > A_05 Preselec. > A_06 Balanceo > A_07 Modelos > A_08/A_09 Preprod. > API + App
    ✅             ✅           ✅           ✅              ⏭️ omitida       ⏭️ omitida      ✅               ✅             ✅
```

**Diferencia con los casos 1 y 2:** técnicamente es el más sencillo, pero el más cargado de negocio. Y el énfasis está en el **despliegue**: una app en Streamlit que llama por POST a una API en FastAPI publicada en Render (diagramas en `docs/flujo_app_api_modelo.png`; el `diagrama_despliegue_v2.png` es el boceto inicial y aún muestra la tasa y plazos de 12 a 48 meses).

---

## 📊 Números del caso

| Dato | Valor |
|---|---|
| Fuente | `prestamos.csv`: 200.000 préstamos, 25 variables |
| Validación | 30 % al azar (60.000), `random_state=42`, en `02_Validacion/prestamos_val.pkl` |
| Entrenamiento | 140.000 préstamos en `train.pkl` |
| Tras calidad | 139.602 préstamos (398 eliminados por ingresos superiores a 400.000 $) |
| Tres modelos | PD (clasificación), EAD (regresión), LGD (regresión) |
| Variables objetivo | Se crean en la transformación a partir de `estado`, `principal`, `imp_amortizado` e `imp_recuperado` |

---

## 1️⃣ Importación · A_01 ✅

> [!QUESTION]- ¿Por qué aquí la validación sí se separa al azar, a diferencia del caso 2?
> Porque cada préstamo es una fila independiente y no hay orden temporal que proteger. En forecasting el orden es la información y un corte al azar enseñaría el futuro al modelo. Aquí no.

* Un solo CSV, sin uniones: `pd.read_csv(ORIGINALES / 'prestamos.csv')`.
* Se reserva el 30 % con `sample(frac=0.3, random_state=42)` y el resto es el conjunto de trabajo.
* La validación no se vuelve a tocar hasta evaluar los modelos.

**En disco:** `02_Validacion/prestamos_val.pkl` · `03_Entrenamiento/train.pkl`

> [!EXAMPLE]- 🏋️ Practica
> Separa un 30 % reproducible de un DataFrame y comprueba que ningún índice de validación quedó dentro del conjunto de trabajo.

---

## 2️⃣ Calidad · A_02 ✅

> [!QUESTION]- `num_meses_desde_ult_retraso` tiene 71.743 nulos y se imputan con cero. ¿Qué riesgo tiene?
> Un cero en esa variable se lee como "hace 0 meses del último retraso", o sea un retraso reciente. Pero aquí el nulo significa "nunca tuvo retraso". El cero cambia el sentido. Por eso conviene revisar su relación con el impago en el EDA, o tratarla con una categoría aparte.

* 0 duplicados. `id_prestamo` está vacía (0 valores): fuera.
* Fuera también `empleo` y `descripcion`: tantos valores distintos que parecen texto libre, escrito por el gestor tal como lo dijo el cliente, sin lista desplegable en el sistema.
* `antigüedad_empleo` nulos como `Desconocido`. Nulos numéricos a cero (sin retrasos, sin tarjetas, sin hipotecas) y recuentos a entero.
* Atípicos: se revisaron `ingresos`, `principal`, `imp_cuota` e `imp_recuperado` a 3,7 desviaciones típicas. Solo se recortó `ingresos` (máximo de 10.999.200): fuera lo superior a 400.000 $, 398 registros, quitados de `cat` y `num` a la vez.

**Puntos abiertos que pasan al EDA:**

| Punto | Cuántos |
|---|---|
| `ingresos` igual a 0 (con `dti` también en 0) | 100 |
| `dti` negativo | 1 |
| `dti` superior a 100 | 163 |
| `porc_uso_revolving` superior a 100 % | 402 |

**En disco:** `train_calidad.pkl` (139.602 filas) + vistas `cat_calidad.pkl` y `num_calidad.pkl` · informe en `06_resultados/Calidad_Datos/`

> [!TIP]- ⚙️ Cómo se hizo
> * **Separar tipos:** `df.select_dtypes(include='string')` para `cat` y `include='number'` para `num`.
> * **Atípicos:** `atipicos_desv_tip()` de `utils.py`. Añadidas ese día al catálogo: `atipicos_cuantiles()` (regla del boxplot), `contar_atipicos()` con elección de método, `convertir_a_int()` y `eliminar_indices()`.
> * **Quitar los mismos registros de cat y num:** `cat, num = eliminar_indices(a_eliminar, cat, num)`.

> [!EXAMPLE]- 🏋️ Practica
> 1. Cuenta los atípicos de `ingresos` con las desviaciones típicas y con los cuantiles. ¿Cuál marca más y por qué?
> 2. Explica sin código por qué eliminar filas de `num` obliga a eliminarlas también de `cat`.

---

## 3️⃣ EDA · A_03 ✅

> [!QUESTION]- ¿Por qué `imp_recuperado` solo tiene valores en los Charged Off, y qué implica para el modelo?
> Porque solo se recupera dinero cuando el banco da el préstamo por perdido. Implica que EAD y LGD se entrenan únicamente con los préstamos en default: fuera de ellos no hay severidad que observar.

* Categóricas: `vivienda` y `finalidad` tienen categorías con muy pocos registros (ANY, NONE, OTHER; `wedding`, `educational`, `renewable_energy`). Se agrupan en la transformación.
* Numéricas: winsorización de `dti` (0 a 100), `num_hipotecas` (hasta 7) y `porc_uso_revolving` (hasta 100).
* La tasa de impago es del 11,9 % (16.549 préstamos). El 38,9 % está en Current, así que la real será algo mayor.
* El cruce de `estado` con `imp_recuperado` es el bivariado clave de esta fase.

**En disco:** `cat_eda.pkl` y `num_eda.pkl` · informe en `06_resultados/EDA/`

> [!EXAMPLE]- 🏋️ Practica
> Calcula la media de `imp_recuperado` por `estado` y explica qué dice sobre cómo entrenar LGD.

---

## 4️⃣ Transformación · A_04 ✅

> [!QUESTION]- ¿Por qué `rating` va de A (0) a G (6) y no al revés?
> Porque con un modelo de coeficientes el número debe significar "más riesgo". Con A en 0 y G en 6 el coeficiente sale positivo y se lee fácil. Con G primero el modelo ajustaría igual, solo con el signo cambiado y más difícil de interpretar.

* Variables objetivo: `target_pd` desde `estado` (y se elimina `estado` para evitar fuga de información), `target_ead` y `target_lgd` en tanto por uno y acotadas entre 0 y 1.
* `num_meses_desde_ult_retraso` se descarta: su tasa de impago es casi igual en todos los tramos.
* OHE para `ingresos_verificados`, `vivienda`, `finalidad` y `num_cuotas`. OE para `antiguedad_empleo` y `rating`, con `unknown_value=12` como red de seguridad para lo que no esté en la lista.
* `num_derogatorios` binaria (84,1 % tiene cero).
* Min Max solo a las ordinales y a las 5 numéricas (desde el 31/10, sin `tipo_interes` ni `imp_cuota`): las columnas OHE ya valen 0 o 1.
* Tres tablones: PD con los 139.602 préstamos; EAD y LGD solo con los 16.549 defaults.

**En disco:** `df_tablon_pd.pkl`, `df_tablon_ead.pkl`, `df_tablon_lgd.pkl` · informe en `06_resultados/Transformacion/`

> [!WARNING]- Lección del ordinal
> Los textos de la lista del OE deben ser idénticos a los de los datos (`Desconocido`, `< 1 year`). Si uno no coincide, `handle_unknown` lo manda al 12 en silencio, sin error. Una primera versión lo hizo y hubo que regenerar los tablones. Tras corregirlo, `Desconocido` es 0 y nadie recibe el 12.

> [!TIP]- ⚙️ Cómo se hizo
> * `OrdinalEncoder(categories=[...], handle_unknown='use_encoded_value', unknown_value=12)`.
> * `Binarizer(threshold=0)` para `num_derogatorios`.
> * Tablones: `pd.concat` con `reset_index(drop=True)` en cada trozo, un `assert` de longitudes e `id_cliente` como índice. Para EAD y LGD, filtrar `target_pd == 1`.

> [!EXAMPLE]- 🏋️ Practica
> 1. Explica con tus palabras qué hace `handle_unknown` en el OE y en el OHE.
> 2. Construye el tablón de EAD desde cero y comprueba que sus identificadores coinciden con los de `target_pd == 1`.

---

## 5️⃣ y 6️⃣ Preselección y balanceo · A_05 y A_06 ⏭️

> [!QUESTION]- ¿Por qué no se hace preselección de variables, si el PD lleva regresión logística?
> Porque la regularización LASSO ya hace una preselección dentro del modelo: lleva a cero los coeficientes que no aportan. Si entran 50 variables y 20 quedan en cero, el modelo trabaja con 30. Y en EAD y LGD el grid compara Ridge, Lasso y HistGradientBoosting (se cambió LightGBM por este último). Además son solo 27 predictoras (29 antes de quitar la tasa y la cuota).

* No se hace preselección ni balanceo (decisión del 24/10/2026). El informe de conexión lo explica en tablas.
* Regla que sí se revisa: solo entran variables que se conocen **al conceder**. Una variable que solo existe después es fuga de información. Ver `Riesgo_y_Banca/Variables_Disponibles_al_Conceder.md` (vault).
* El caso es de **adquisición**, no de comportamiento: ver `Riesgo_y_Banca/Modelos_Adquisicion_vs_Comportamiento.md` (vault).

**En disco:** informe en `06_resultados/Preseleccion/`

> [!EXAMPLE]- 🏋️ Practica
> 1. Explica con un ejemplo de 50 variables qué hace LASSO y por qué las variables deben ir escaladas.
> 2. Para cada predictora del tablón, responde: ¿la tendría la API cuando llega un cliente nuevo?

---

## 7️⃣ a 9️⃣ Modelización, preproducción y scoring ✅

> [!QUESTION]- ¿Por qué el artefacto final no se evalúa con la reserva?
> Porque se reentrena con todos los datos, reserva incluida, y ya la vio. La nota válida es la de la fase 1: entrenar con el 70 %, medir una sola vez con la reserva, y luego reentrenar con la misma receta para producir. Resumen de una página: `01_Documentos/Flujo_Preproduccion_a_Produccion.pdf`.

| Modelo | Algoritmo | Métrica en la reserva |
|---|---|---|
| PD | Regresión logística ElasticNet (en el notebook, Lasso) | AUC 0,6979 |
| EAD | Ridge | MAE 0,1647 |
| LGD | HistGradientBoosting (en el notebook, Lasso) | MAE 0,0900 |

> [!WARNING] Reentrenado el 31/10/2026 sin `tipo_interes` ni `imp_cuota`
> Antes: AUC 0,7006, EAD 0,1588, LGD 0,0900 (con tasa y cuota). La tasa la fija el banco según el riesgo: usarla para predecir el riesgo es circular (correlación 0,953 con el rating y signo negativo en el Lasso). Quitarla costó casi nada (AUC de validación cruzada 0,7002 a 0,6976, ruido de ±0,005). Sin la tasa, los candidatos empatan: por eso gana un modelo lineal en EAD y el LGD sigue siendo casi constante (89 %).

* El notebook 10 quita la reserva al cargar el CSV y trabaja solo con el 70 %. Su 70/30 interno solo comprueba que reproduce lo de 07, 08 y 09.
* `01_reentrenamiento.py` busca hiperparámetros con validación cruzada, mide en la reserva y reentrena con todo. `02_produccion_scoring.py` puntúa con los tres artefactos y calcula la pérdida esperada.
* Informes: `06_resultados/Modelizacion/informe_modelizacion.md` y `06_resultados/Preproduccion/informe_preproduccion.md`.

**En disco:** `07_despliegue/artefacto_pipeline_{pd,ead,lgd}.pkl` · `07_despliegue/pre-produccion/` (manifiesto y resumen)

> [!EXAMPLE]- 🏋️ Practica
> 1. Explica sin código por qué el escalado se ajusta dentro de cada pliegue de la validación cruzada.
> 2. Calcula a mano la pérdida esperada de un préstamo con PD 10 %, principal 10.000, EAD 0,7 y LGD 0,9.

---

## 🔟 API y app · A_11 y A_12 ✅

> [!QUESTION]- ¿Por qué la app no calcula nada y todo lo pide a la API?
> Porque así otras aplicaciones (un CRM, un ERP) pueden usar la misma API sin pasar por nuestra app. La app solo pinta lo que la API devuelve y aplica el semáforo. En la vida real el historial del cliente vendría de una base protegida; aquí se simula con un CSV ficticio de 50 clientes.

| Pieza | Dónde | Qué hace |
|---|---|---|
| API | https://zero3-riesgos.onrender.com/ | `GET /health`, `GET /clientes` (solo ids), `GET /clientes/{id}` (ficha) y `POST /predict` (lista de registros con 5 campos; devuelve PD, EAD, LGD, pérdida esperada y tasa sugerida) |
| App | https://zero3-riesgos-aplicacion.onrender.com | Marca Bancandrex. Pestañas "Evaluar solicitud" y "Comparar plazos" (36 contra 60 meses). Ficha del cliente, banner de riesgo con la tasa sugerida, medidor de PD, barras de EAD y LGD, cascada de la pérdida esperada |
| URL de la API en la app | Variable de entorno `API_BASE_URL` | Por defecto `http://127.0.0.1:8000`; en Render vale la URL de la API. El código no cambia |
| Python en Render | `PYTHON_VERSION=3.12.13` | Los `.pkl` solo cargan bien con la misma versión menor de Python |

* **Hallazgo con la tasa (resuelto el 31/10):** con la tasa como predictora, los coeficientes eran `rating` +4,47, `imp_cuota` +2,32 y `tipo_interes` −2,46: variables que se explican entre sí, con signos opuestos, y subir la tasa bajaba la PD. Se quitaron `tipo_interes` e `imp_cuota`; la tasa es ahora una respuesta.
* **Tasa sugerida** = 3,0 (fondeo) + 0,7 (costos) + 1,3 (margen) + 100 × pérdida esperada / principal / años. Supuestos ilustrativos; es una tasa mínima, sin pérdida inesperada ni capital. Para un cliente de rating A sale cerca de 5,7 %.
* El semáforo se calcula con la PD. Con los modelos nuevos y los 200.000 préstamos puntuados, 8 % y 15 % dejan 38,0 % en bajo, 37,2 % en medio y 24,8 % en alto.
* En el README quedaron escritos los supuestos: el cliente ya existe, monto de 500 a 40.000, plazos 36 y 60 (los únicos de los datos), tope de ingresos de 400.000 y las limitaciones de esta primera prueba.

**En disco:** `07_despliegue/API/` y `07_despliegue/app/` · informes en `06_resultados/Despliegue_API/` y `06_resultados/Despliegue_App/` · diagramas en `docs/flujo_app_api_modelo.png` y `docs/glosario_schemas_scoring_main.png`

> [!EXAMPLE]- 🏋️ Practica
> 1. Explica sin código qué recibe y qué devuelve `POST /predict`, y por qué `/clientes` no manda el historial.
> 2. Explica por qué la tasa no puede ser una entrada del modelo de riesgo, y cómo se calcula la tasa sugerida.

---

## 🧰 Funciones del catálogo usadas hasta ahora

| Función | Para qué se usó |
|---|---|
| `atipicos_desv_tip()` | Atípicos por desviaciones típicas |
| `atipicos_cuantiles()` | Alternativa por cuantiles (nueva) |
| `contar_atipicos()` | Contar atípicos por método (ampliada) |
| `convertir_a_int()` | Pasar recuentos a entero (nueva) |
| `eliminar_indices()` | Quitar los mismos registros de varios DataFrames (nueva) |
| `plot_roc_multi_np()` | Curvas ROC de varios modelos con su AUC (fase 07) |
| `reporting_regresion()` | Reporte de un regresor en una llamada: real contra predicho, residuos y error por tramos (fases 08 y 09) |

---

## 📝 Pendientes que no son de fase

* Informe corto de arquitectura con dos diagramas (vida real frente a simulación, y cómo se consume la API).
* `pyproject.toml` sin las dependencias de API y app: `uv add streamlit plotly requests fastapi uvicorn pydantic cloudpickle`.
* Leer la normativa de Basilea (matiz entre pérdida esperada y capital regulatorio, ver `Riesgo_y_Banca/Basilea_y_Crisis_2008.md`).
* Proyecto pequeño sobre `empleo` y `descripcion`: normalizar el texto y agrupar los empleos en categorías, a ver si dan variables útiles.
* Llevar a `docs/` y a `Recursos-Assets` los diagramas de ChatGPT (vida real frente a simulación, y cómo se consume la API). Los de flujo app, API y modelo y el glosario de schemas se rehicieron el 31/10 (sin tasa ni cuota): están en `docs/` y falta copiarlos a `Recursos-Assets`.
* Los 100 préstamos con `ingresos` igual a 0 y el motivo de dejar fuera `num_hipotecas`, `porc_tarjetas_75p` y `num_cancelaciones_12meses`.
