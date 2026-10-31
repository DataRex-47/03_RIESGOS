# Resumen de limpieza y preproducción (A_08)

Proyecto Scoring de Riesgos, Caso 3. Auditoría del notebook finalista construido a partir de
los notebooks de desarrollo A_01..A_07.

- Notebook finalista: [10_Preproduccion.ipynb](../../03_notebooks/10_Preproduccion.ipynb)
- Manifiesto para A_09: [00_manifiesto_preproduccion.json](./00_manifiesto_preproduccion.json)
- Informe de modelización de referencia: [informe_modelizacion.md](../../06_resultados/Modelizacion/informe_modelizacion.md)

---

## Fase 0. Mapeo del proyecto

| Elemento | Ruta verificada |
|---|---|
| Datos originales | `02_datos/01_Originales/prestamos.csv` |
| Datos intermedios | `02_datos/02_Validacion/`, `02_datos/03_Entrenamiento/` |
| Notebooks | `03_notebooks/01..10` |
| Utilidades | `04_scripts/utils.py` |
| Modelos | `05_modelos/` (vacío: la serialización es de A_09) |
| Resultados | `06_resultados/` |
| Despliegue | `07_despliegue/` |

**Fuente de datos.** En [01_Importacion_Datos.ipynb](../../03_notebooks/01_Importacion_Datos.ipynb)
hay una única lectura de CSV:

```python
datos = pd.read_csv(ORIGINALES / 'prestamos.csv')
```

Se lee con el separador por defecto (`,`), sin `sep` ni `encoding` explícitos. El notebook
finalista la replica como `df = pd.read_csv(ORIGINALES / 'prestamos.csv')`.

Columnas del CSV (25): `id_cliente`, `empleo`, `antigüedad_empleo`, `ingresos`,
`ingresos_verificados`, `rating`, `dti`, `vivienda`, `num_hipotecas`, `num_lineas_credito`,
`porc_tarjetas_75p`, `porc_uso_revolving`, `num_cancelaciones_12meses`, `num_derogatorios`,
`num_meses_desde_ult_retraso`, `id_prestamo`, `descripcion`, `finalidad`, `principal`,
`tipo_interes`, `num_cuotas`, `imp_cuota`, `imp_amortizado`, `estado`, `imp_recuperado`.

**Desviación de nomenclatura:** las instrucciones del agente hablan de `08_Preproduccion.ipynb`.
El proyecto evolucionó a `07_PD`, `08_EAD`, `09_LGD`, y el usuario creó `10_Preproduccion.ipynb`
con las secciones ya definidas. El entregable se construye sobre ese notebook y no se crea un
`08_Preproduccion.ipynb` paralelo.

---

## Fase 1. Integración lineal

Se recorrieron, en orden, las celdas de código de A_01..A_07 y se copió el **código real**
transformador. En los notebooks de origen esas transformaciones ocupaban más de 30 celdas; en el
notebook finalista quedan agrupadas en tres celdas (calidad, EDA e ingeniería de variables) más una
celda por modelo, además de la celda de imports/rutas y la de carga del CSV que ya existían en el
esqueleto del usuario.

| Origen | Pasos integrados |
|---|---|
| 01_Importacion_Datos | lectura del CSV |
| 02_Calidad_Datos | `reset_index`, borrado de `id_prestamo`, split `cat`/`num`, borrado de `empleo` y `descripcion`, imputación de la antigüedad, `num.fillna(0)`, recuentos a `int`, filtro `ingresos > 400000` y borrado de esos registros |
| 03_EDA | renombrado `antigüedad_empleo` → `antiguedad_empleo`, recortes de `dti`, `num_hipotecas` y `porc_uso_revolving` |
| 04_Transformacion_Datos | `target_pd`, `pendiente`, `target_ead`, `target_lgd`, recortes a [0, 1], unificación de categorías raras, borrado de `num_meses_desde_ult_retraso`, One Hot, Ordinal, Binarizer, Min Max, concatenación de features y construcción de los tres tablones |
| 07_Modelizacion_Clasificacion_PD | reserva 70/30 estratificada y regresión logística Lasso (`saga`, `l1_ratio=1`, `C=0.05`, `max_iter=5000`) |
| 08_Modelizacion_Regresion_EAD | reserva 70/30, HistGradientBoosting (`lr=0.05`, `l2=0.25`, `depth=5`, `iter=200`, `leaf=60`) y recorte de la predicción |
| 09_Modelizacion_Regresion_LGD | reserva 70/30, HistGradientBoosting (`lr=0.01`, `l2=0`, `depth=5`, `iter=50`, `leaf=60`) y recorte de la predicción |

**Observaciones de la integración**

- Las celdas de inspección (`.head()`, `.info()`, `.describe()`, `value_counts`, gráficos,
  correlaciones, gain/lift, análisis de umbrales) no son transformaciones y no se integran.
- El resto de las celdas de 04 son análisis de `num_meses_desde_ult_retraso`; solo se conserva la
  línea que borra la columna.
- En 02 la celda que borra los textos está corrupta:
  `cat.drop(columns=['empleo',], inplace 'descripcion'=True)`. Se escribe con la intención
  documentada en su propio comentario: `cat.drop(columns=['empleo', 'descripcion'], inplace=True)`.
- Duplicidad eliminada respecto a 04: los bloques `variables_ead` y `variables_lgd` eran copias
  exactas de `X_pd`. Se calcula una sola vez `variables` y se reutiliza en los tres tablones.
- En 04, `target_pd`, `y_ead` y `y_lgd` se extraían con `reset_index(drop=True)`; se mantiene porque
  `cat` y `num` conservan el índice original tras el filtro de ingresos.
- Se conservan los `assert` de longitud e índice que ya existían en 04: validan el contrato de los
  tablones sin coste de ejecución.

---

## Fase 2. Limpieza y enfoque en el modelo final

El notebook finalista mantiene solo lo que alimenta a las features finalistas, a las variables
objetivo y a los tres modelos.

**Se conserva**

1. Carga del CSV original.
2. Calidad: borrado de `id_prestamo`, `empleo`, `descripcion`; imputación de la antigüedad; nulos
   numéricos a cero; recuentos a entero; borrado de los préstamos con ingresos > 400.000.
3. EDA: renombrado de la antigüedad y recortes de `dti`, `num_hipotecas` y `porc_uso_revolving`.
4. Ingeniería: `target_pd`, `target_ead`, `target_lgd`, unificación de categorías raras, One Hot,
   Ordinal, Binarizer, Min Max, tablón PD y tablones EAD/LGD.
5. La reserva de 01 (`prestamos_val.pkl`) se quita justo después de leer el CSV; todo lo demás trabaja con el resto.
6. Modelo final de cada una de las tres pérdidas, entrenado con ese 70 % y los hiperparámetros ganadores de A_07.

**Sobre la reserva de validación.** Es la misma que se apartó en 01, leída de su archivo. El notebook
no la usa ni evalúa: la medición con ella la hace A_09 en su fase 1. El 30 % reservado **no se usa para elegir hiperparámetros** (ya vienen fijados de A_07):
solo se predice sobre él y se mide. Cuando A_09 construya el pipeline de producción, el
entrenamiento se hará sobre el 100 % del tablón; ese cambio es de A_09, no de este notebook.

**Se descarta y por qué**

| Descartado | Motivo |
|---|---|
| `GridSearchCV` y los grids de 07/08/09 | Fase de experimentación: los hiperparámetros ya están elegidos |
| Lectura de `train.pkl`, `cat_calidad.pkl`, `num_calidad.pkl`, `cat_eda.pkl`, `num_eda.pkl` | Se sustituyen por la cadena completa desde el CSV |
| `to_pickle` de tablones y datasets intermedios | La serialización de artefactos es de A_09 |
| `num_hipotecas`, `porc_tarjetas_75p`, `num_cancelaciones_12meses`, `num_meses_desde_ult_retraso` | Fuera del tablón por decisión de fases anteriores (el recorte y el `astype` se conservan para no alterar el flujo) |
| `estado`, `imp_amortizado`, `imp_recuperado`, `pendiente` | Fuga de información: no se conocen al conceder el préstamo |
| Gráficos, gain/lift, tablas de coeficientes e informes | No aportan a la generación de features ni al ajuste final |

**Features finalistas: 29** (19 de One Hot + 9 de Min Max + 1 binaria), con `id_cliente` como índice.

**Modelos finales**

| Modelo | Clase | Hiperparámetros | Métrica sobre la validación |
|---|---|---|---|
| PD | `LogisticRegression` | `solver='saga'`, `l1_ratio=1`, `C=0.05`, `max_iter=5000` | AUC 0,695 |
| EAD | `HistGradientBoostingRegressor` | `learning_rate=0.05`, `l2_regularization=0.25`, `max_depth=5`, `max_iter=200`, `min_samples_leaf=60`, `scoring='neg_mean_absolute_error'` | MAE 0,158 |
| LGD | `HistGradientBoostingRegressor` | `learning_rate=0.01`, `l2_regularization=0`, `max_depth=5`, `max_iter=50`, `min_samples_leaf=60`, `scoring='neg_mean_absolute_error'` | MAE 0,089 |

**Postproceso de las predicciones:** EAD y LGD se recortan a [0, 1] con `np.where`, igual que en
08 y 09, y el MAE se mide sobre la predicción ya recortada. PD devuelve probabilidad, que ya cae
en [0, 1].

---

## Volumetría observada en la última ejecución

El notebook lee el CSV, quita la reserva de 01 y trabaja solo con el resto:

| Paso | Forma |
|---|---|
| CSV original / reserva de 01 / trabajo | (200.000, 25) / (60.000, 25) / (140.000, 25) |
| `cat` / `num` tras calidad y filtro de ingresos | (139.602, 7) / (139.602, 15) |
| `tabla_pd` / `tabla_ead` / `tabla_lgd` | (139.602, 30) / (16.549, 31) / (16.549, 31) |

Son 139.602 préstamos y 16.549 impagos (11,9 %), los mismos de la fase de desarrollo. La métrica de
la tabla de arriba sale del 70/30 interno del notebook, que solo comprueba que se reproducen 07, 08 y 09.

---

## Puntos abiertos que arrastra el proyecto

- 100 préstamos con `ingresos` igual a 0: se mantienen.
- El `drop` de `empleo`/`descripcion` estaba corrupto en 02 y se corrigió según el comentario de la celda.
- El modelo de LGD predice prácticamente un valor fijo (0,888): es un límite de los datos, no del código.

## Siguiente paso

A_09_Pipelines debe consumir [00_manifiesto_preproduccion.json](./00_manifiesto_preproduccion.json)
y construir los pipelines de sklearn con los pasos del DAG (`P01`..`P30`).
