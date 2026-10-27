---
tags: [bitacora, scoring-riesgos, caso-ml, contexto]
aliases: [Bitácora Scoring de Riesgos, Contexto del Proyecto 3]
date: 2026-10-11
updated: 2026-10-24
status: 🌱 (Seed)
---

# 🧭 Bitácora Maestra: Caso 3 · Scoring de Riesgos

> [!SUMMARY] Para qué sirve
> Reengancharte en 3 minutos y practicar sin releer los notebooks.
> Cada fase trae: pregunta de recall, qué pasó, qué quedó en disco, cómo se hizo y un ejercicio.
> **El contexto del caso (negocio, fórmula, sectores) no está aquí**: está en el vault, en `Casos_ML/Caso_03_Scoring_Riesgos.md` y en la carpeta `Riesgo_y_Banca/`. Aquí solo va el seguimiento. La guía corta en PDF está en esta misma carpeta: `Scoring de Riesgos Guia.pdf`.

---

## 📍 Dónde estoy

**Fases 01 a 04 cerradas (17/10/2026).** Importación, Calidad, EDA y Transformación hechas. Los tres tablones (PD, EAD y LGD) están en `03_Entrenamiento`. **Fases 05 y 06 omitidas por decisión (24/10/2026):** no hay preselección ni balanceo, queda un informe de conexión en `06_resultados/Preseleccion/`.

**Siguiente acción:** modelización (notebook 07): PD con regresión logística y LASSO, EAD y LGD con LightGBM.

> [!IMPORTANT] 📌 Para releer: la idea que lo ordena todo
> **Valor esperado = probabilidad de que ocurra un evento × impacto económico si ocurre.**
> En banca se parte el impacto en dos: **Pérdida esperada = PD × Principal × EAD × LGD**, con tres modelos (una clasificación y dos regresiones). En otros sectores son dos modelos. Detalle: `Riesgo_y_Banca/Valor_Esperado_Cualquier_Negocio.md` (vault).

---

## 🔗 La cadena

```
A_01 Import > A_02 Calidad > A_03 EDA > A_04 Transform > A_05 Preselec. > A_06 Balanceo > A_07 Modelos > A_08/A_09 Preprod. > API + App
    ✅             ✅           ✅           ✅              ⏭️ omitida       ⏭️ omitida      📍 AQUÍ          ⬜             ⬜
```

**Diferencia con los casos 1 y 2:** técnicamente es el más sencillo, pero el más cargado de negocio. Y el énfasis está en el **despliegue**: una app en Streamlit que llama por POST a una API en FastAPI publicada en Render (diagrama en `docs/diagrama_despliegue_v2.png`).

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
* Min Max solo a las ordinales y a las 7 numéricas: las columnas OHE ya valen 0 o 1.
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
> Porque la regularización LASSO ya hace una preselección dentro del modelo: lleva a cero los coeficientes que no aportan. Si entran 50 variables y 20 quedan en cero, el modelo trabaja con 30. Y LightGBM, que se usa en EAD y LGD, elige variables al partir sus árboles. Además son solo 29 predictoras.

* No se hace preselección ni balanceo (decisión del 24/10/2026). El informe de conexión lo explica en tablas.
* Regla que sí se revisa: solo entran variables que se conocen **al conceder**. Una variable que solo existe después es fuga de información. Ver `Riesgo_y_Banca/Variables_Disponibles_al_Conceder.md` (vault).
* El caso es de **adquisición**, no de comportamiento: ver `Riesgo_y_Banca/Modelos_Adquisicion_vs_Comportamiento.md` (vault).

**En disco:** informe en `06_resultados/Preseleccion/`

> [!EXAMPLE]- 🏋️ Practica
> 1. Explica con un ejemplo de 50 variables qué hace LASSO y por qué las variables deben ir escaladas.
> 2. Para cada predictora del tablón, responde: ¿la tendría la API cuando llega un cliente nuevo?

---

## 🧰 Funciones del catálogo usadas hasta ahora

| Función | Para qué se usó |
|---|---|
| `atipicos_desv_tip()` | Atípicos por desviaciones típicas |
| `atipicos_cuantiles()` | Alternativa por cuantiles (nueva) |
| `contar_atipicos()` | Contar atípicos por método (ampliada) |
| `convertir_a_int()` | Pasar recuentos a entero (nueva) |
| `eliminar_indices()` | Quitar los mismos registros de varios DataFrames (nueva) |

---

## 📝 Pendientes que no son de fase

* README del proyecto: se escribe cuando exista la app, con la estructura del de Forecasting más una sección para los tres modelos y el diagrama de despliegue.
* Leer la normativa de Basilea (matiz entre pérdida esperada y capital regulatorio, ver `Riesgo_y_Banca/Basilea_y_Crisis_2008.md`).
* Proyecto pequeño sobre `empleo` y `descripcion`: normalizar el texto y agrupar los empleos en categorías, a ver si dan variables útiles.
