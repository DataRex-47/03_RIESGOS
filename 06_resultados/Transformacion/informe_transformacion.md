# Informe de transformación de datos

| | Entrada | Salida |
|---|---|---|
| Datos | `cat_eda.pkl` y `num_eda.pkl`: 139.602 préstamos, 7 variables categóricas y 15 numéricas | 3 tablones: PD, EAD y LGD, con 27 predictoras cada uno |
| Referencia | `01_Documentos/FaseDesarrollo_PlantillaTransformaciones.xlsx` | `06_resultados/Transformacion/` |

## 1. Qué se hizo con cada variable

| Variable | Transformación | Por qué | Resultado |
|---|---|---|---|
| `rating` | Ordinal de A (0) a G (6), luego Min Max | Tiene un orden natural | 1 columna, de 0 a 1 |
| `antiguedad_empleo` | Ordinal de `Desconocido` (0) a `10+ years` (11), luego Min Max | Tiene un orden natural | 1 columna, de 0 a 1 |
| `ingresos_verificados` | One Hot Encoding | Sus 3 categorías no tienen un orden claro | 3 columnas de 0 o 1 |
| `vivienda` | ANY, NONE y OTHER (70 préstamos) pasan a MORTGAGE, luego OHE | Categorías casi vacías | 3 columnas de 0 o 1 |
| `finalidad` | `wedding`, `educational` y `renewable_energy` (255 préstamos) pasan a `other`, luego OHE | Categorías casi vacías | 11 columnas de 0 o 1 |
| `num_cuotas` | One Hot Encoding | Solo tiene 2 valores: 36 y 60 meses | 2 columnas de 0 o 1 |
| `num_derogatorios` | Binarizada: 0 si no hay incidencias, 1 si hay alguna | El 84,1 % tiene 0 y la cola llega a 46 con muy pocos casos | 117.439 con 0 y 22.163 con 1 |
| `ingresos`, `dti`, `num_lineas_credito`, `porc_uso_revolving`, `principal` | Min Max | Dejarlas entre 0 y 1, igual que el resto | 5 columnas, de 0 a 1 |
| `tipo_interes`, `imp_cuota` | Descartadas (decisión del 31/10/2026) | La tasa la fija el banco a partir del riesgo del cliente, así que usarla para predecir ese riesgo es circular. Casi repite el rating (correlación 0,953) y la cuota se calcula con la tasa | Fuera del tablón. La tasa pasa a ser una respuesta de la API (tasa sugerida) |
| `num_meses_desde_ult_retraso` | Descartada | La tasa de impago es de 11 % a 13 % en todos los tramos, sin relación clara | Fuera del tablón |
| `num_hipotecas`, `porc_tarjetas_75p`, `num_cancelaciones_12meses` | Sin transformación | La plantilla no las incluye y no dice el motivo | Fuera del tablón |
| `estado` | Se usa para crear `target_pd` y después se elimina | Evita fuga de información y en producción no se conoce | Fuera del tablón |
| `imp_amortizado`, `imp_recuperado` | Solo para crear las variables objetivo | Reflejan lo ocurrido después de dar el préstamo | Fuera del tablón |
| `id_cliente` | Pasa a ser el índice | Sirve para rastrear el préstamo, no para predecir | Índice |

Total de predictoras: 19 columnas OHE, 7 con Min Max y 1 binaria, es decir 27.

## 2. Un préstamo antes y después

Préstamo 4798121, que terminó en Charged Off:

| Variable | Valor original | Valor en el tablón |
|---|---|---|
| `rating` | D | 0,50 (código 3 de 6) |
| `antiguedad_empleo` | 10+ years | 1,00 (código 11) |
| `ingresos_verificados` | Not Verified | `Not Verified` = 1, las otras dos = 0 |
| `vivienda` | RENT | `RENT` = 1, las otras dos = 0 |
| `finalidad` | debt_consolidation | `debt_consolidation` = 1, las otras 10 = 0 |
| `num_cuotas` | 36 months | `36 months` = 1, `60 months` = 0 |
| `num_derogatorios` | 0 | 0 |
| `ingresos` | 65.000 | 0,16 |
| `dti` | 25,4 | 0,25 |
| `porc_uso_revolving` | 86,2 | 0,86 |
| `principal` | 10.000 | 0,24 |
| `estado` | Charged Off | `target_pd` = 1 |

## 3. Variables objetivo

| Objetivo | Cómo se calcula | Ajustes | En los defaults |
|---|---|---|---|
| `target_pd` | 1 si `estado` es Charged Off, "Does not meet the credit policy. Status:Charged Off" o Default, y 0 en el resto | Ninguno | 16.549 préstamos, el 11,9 % del total |
| `target_ead` | (`principal` menos `imp_amortizado`) entre `principal` | 14 valores apenas negativos pasan a 0 | Media 0,70 |
| `target_lgd` | 1 menos (`imp_recuperado` entre `pendiente`) | Pendiente 0 da nulo y se pone 0 (66.731 préstamos, ninguno en default). 56 valores negativos pasan a 0 | Media 0,89 |

EAD y LGD van en tanto por uno para acotarlos entre 0 y 1 y comparar préstamos de distinto tamaño. Para volver a dinero se multiplica la predicción por el `principal`.

## 4. Cómo quedaron los códigos ordinales

| `rating` | Código | Valor escalado | Impago |
|---|---|---|---|
| A | 0 | 0,00 | 3 % |
| B | 1 | 0,17 | 8 % |
| C | 2 | 0,33 | 13 % |
| D | 3 | 0,50 | 19 % |
| E | 4 | 0,67 | 27 % |
| F | 5 | 0,83 | 34 % |
| G | 6 | 1,00 | 37 % |

| `antiguedad_empleo` | Código | Préstamos | Impago |
|---|---|---|---|
| `Desconocido` | 0 | 9.047 | 15 % |
| `< 1 year` | 1 | 11.863 | 12 % |
| `1 year` a `9 years` | 2 a 10 | 72.751 | 11 % a 13 % |
| `10+ years` | 11 | 45.941 | 11 % |

## 5. Unificar datasets: qué dataframes se unen

Piezas que se crean en el notebook:

| DataFrame | Qué contiene | Columnas | Se crea con |
|---|---|---|---|
| `cat_ohe` | Variables categóricas con OHE | 19 | `OneHotEncoder` sobre `cat` |
| `cat_oe` | `antiguedad_empleo` y `rating` como códigos | 2 | `OrdinalEncoder` sobre `cat` |
| `num_escalar` | Las 5 numéricas a escalar, sin escalar | 5 | Selección de `num` |
| `df_res` | `cat_oe` y `num_escalar` juntos | 7 | `pd.concat` |
| `df_mms` | `df_res` ya escalado, con sufijo `_scaled` | 7 | `MinMaxScaler` |
| `num_bin` | `num_derogatorios` binaria | 1 | `Binarizer` |
| `id_cliente` | Identificador del préstamo | 1 | Columna de `num` |
| `y_pd`, `y_ead`, `y_lgd` | Las variables objetivo | 1 cada una | `cat` y `num` |

Cómo se unen en cada tablón:

| Tablón | Se une así | Filtro | Resultado |
|---|---|---|---|
| `tabla_pd` | `id_cliente` + `X_pd` (`cat_ohe` + `df_mms` + `num_bin`) + `y_pd` | Ninguno | 139.602 filas y 28 columnas |
| `tabla_ead` | `id_cliente` + `variables_ead` (`cat_ohe` + `df_mms` + `num_bin`) + `target_pd` + `y_ead` | `target_pd` igual a 1 | 16.549 filas y 29 columnas |
| `tabla_lgd` | `id_cliente` + `variables_lgd` (`cat_ohe` + `df_mms` + `num_bin`) + `target_pd` + `y_lgd` | `target_pd` igual a 1 | 16.549 filas y 29 columnas |

En los tres casos:

- Cada pieza lleva `reset_index(drop=True)` para unirse por posición.
- Un `assert` comprueba que todas las longitudes coincidan.
- `id_cliente` pasa a ser el índice con `set_index`.
- En EAD y LGD se separa después `X_ead` y `X_lgd` quitando `target_pd` y el objetivo.

## 6. Decisiones clave

- **Orden de menor a mayor.** Con A en 0 y G en 6, el coeficiente del rating sale positivo en un modelo como la regresión logística: más número, más riesgo. Con G primero el ajuste sería el mismo y solo cambiaría el signo, más difícil de leer.
- **Min Max solo a ordinales y numéricas.** Las 19 columnas OHE ya valen 0 o 1, así que no necesitan escalarse. Así todas las predictoras quedan en el mismo rango y ninguna pesa más por su escala.
- **El 12 de `handle_unknown`.** En el ordinal, `unknown_value=12` hace que un valor fuera de la lista no dé error en producción. En el OHE, `ignore` deja esas columnas en 0. Con los datos de entrenamiento nadie recibe el 12, siempre que los textos de la lista sean idénticos a los de los datos.
- **EAD y LGD solo con defaults.** Miden la severidad de un impago, que solo se observa cuando el impago ocurre.

## 7. Resultado

| Tablón | Filas | Predictoras | Objetivo | Para el modelo |
|---|---|---|---|---|
| `df_tablon_pd.pkl` | 139.602 | 27 | `target_pd` | PD |
| `df_tablon_ead.pkl` | 16.549 | 27 | `target_ead` | EAD |
| `df_tablon_lgd.pkl` | 16.549 | 27 | `target_lgd` | LGD |

Los tres están en `02_datos/03_Entrenamiento/`. EAD y LGD conservan `target_pd` (siempre 1) solo para filtrar, y hay que quitarla antes de entrenar.

## 8. Pendientes

- 100 préstamos con `ingresos` igual a 0, que siguen sin tratar.
- Anotar el motivo de dejar fuera `num_hipotecas`, `porc_tarjetas_75p` y `num_cancelaciones_12meses`.
- Si el modelo lo necesita, quitar una categoría por variable en el OHE.
