# Informe de preproducción y despliegue de los modelos

Caso 3, Scoring de Riesgos. Este informe cuenta qué pasó entre el notebook de modelización y los scripts de producción, y explica cómo leer cada script para no perderse al volver a ellos.

## 1. El flujo completo en una tabla

| Paso | Archivo | Qué hace | Qué produce |
|---|---|---|---|
| 1 | `01_Importacion_Datos` | Aparta el 30 % de los clientes como reserva | `prestamos_val.pkl` y `train.pkl` |
| 2 | Notebooks 02 a 09 | Desarrollo y elección de modelos con el 70 % | Los hiperparámetros ganadores |
| 3 | `10_Preproduccion.ipynb` | Junta todo el proceso desde el CSV y entrena los tres modelos | Manifiesto y resumen de limpieza |
| 4 | `01_reentrenamiento.py` | Busca hiperparámetros, mide con la reserva y reentrena con todo | Tres artefactos `.pkl` |
| 5 | `02_produccion_scoring.py` | Carga los artefactos y puntúa préstamos | CSV con scores y pérdida esperada |

## 2. Qué pasó en el notebook 10

La reserva de 01 se lee de su archivo y se quita justo después de cargar el CSV. Desde ahí todo el notebook trabaja solo con el 70 % restante, así que ningún paso ajusta con datos de la reserva.

| Paso | Filas |
|---|---|
| CSV original | 200.000 |
| Reserva de 01 | 60.000 |
| Trabajo (70 %) | 140.000 |
| Tras calidad y filtro de ingresos mayores a 400.000 | 139.602 |
| Tablón de PD | 139.602 filas y 30 columnas |
| Tablones de EAD y LGD (solo impagos) | 16.549 filas y 31 columnas |

Cada modelo hace un 70/30 interno solo para comprobar que el notebook reproduce lo que ya habías obtenido en 07, 08 y 09. No sirve para elegir hiperparámetros.

| Modelo | Métrica del notebook | Referencia en 07, 08 y 09 |
|---|---|---|
| PD (AUC) | 0,695 | 0,698 a 0,703 |
| EAD (MAE) | 0,158 | 0,158 |
| LGD (MAE) | 0,089 | 0,088 a 0,090 |

## 3. Qué pasó en el reentrenamiento

El script trabaja en dos fases por cada modelo. La fase 1 mide: busca hiperparámetros con validación cruzada dentro del 70 % y evalúa una sola vez con la reserva. La fase 2 produce: reentrena la mejor receta con todos los datos y guarda el artefacto.

| Modelo | Mejores hiperparámetros | Validación cruzada | Reserva de 01 |
|---|---|---|---|
| PD, regresión logística Lasso | `l1_ratio` 1, `C` 1 | AUC 0,6999 | AUC 0,7006 |
| EAD, HistGradientBoosting | `learning_rate` 0,05, `max_depth` 20, `max_iter` 200, `l2_regularization` 0,5 | MAE 0,1577 | MAE 0,1588 |
| LGD, HistGradientBoosting | `learning_rate` 0,05, `max_depth` 5, `max_iter` 50, `l2_regularization` 0,25 | MAE 0,0887 | MAE 0,0900 |

Los resultados en validación cruzada y en la reserva son casi iguales, así que no hay sobreajuste. La nota que se reporta de cada modelo es la de la reserva. El artefacto final se entrenó con todos los datos y ya no se puede evaluar con la reserva, porque ya la vio.

| Artefacto | Filas con las que se entrenó |
|---|---|
| `artefacto_pipeline_pd.pkl` | 199.433 préstamos |
| `artefacto_pipeline_ead.pkl` | 23.644 impagos |
| `artefacto_pipeline_lgd.pkl` | 23.644 impagos |

Los hiperparámetros ganadores no son exactamente los del notebook (por ejemplo `C` 1 en vez de 0,05 en PD). Las métricas son prácticamente iguales, porque las curvas son planas cerca del óptimo.

## 4. Cómo leer `01_reentrenamiento.py`

Se lee de arriba hacia abajo en seis bloques.

| Bloque | Qué hace | Para qué sirve |
|---|---|---|
| Rutas y listas de columnas | Define rutas, estados de default y qué columnas reciben qué tratamiento | Es el único lugar que hay que tocar si cambian los datos |
| Funciones de fila y de reglas | `prepara_datos` quita ingresos mayores a 400.000; `unifica_categorias` y `limita_numericas` aplican las reglas de 03 y 04 | Repite las decisiones del notebook como piezas reutilizables |
| `crea_objetivos` | Calcula PD, EAD y LGD a partir de estado, principal, amortizado y recuperado | Solo se usa para entrenar y medir, nunca entra como variable del modelo |
| `crea_preprocesador` | Cuatro tratamientos por grupo: One Hot a las categóricas, orden y escala a antigüedad y rating, recorte y escala a las numéricas, 0 o 1 a derogatorios | Se ajusta dentro de cada pipeline, así que cada pliegue aprende solo de su entrenamiento |
| `crea_pipeline` y espacios de búsqueda | Une preprocesador y modelo base. PD prueba 18 combinaciones; EAD y LGD mezclan Ridge, Lasso e HistGradientBoosting y prueban 20 al azar | Es la lista de candidatos que compiten |
| Reserva y bucle por modelo | Separa la reserva por `id_cliente`; para PD, EAD y LGD arma X e y, busca, mide en la reserva, reentrena con todo y guarda el `.pkl` | Es el corazón del script: las fases 1 y 2 |

Dos detalles para recordar. EAD y LGD se entrenan solo con préstamos en default, y se aplican a todos los clientes al puntuar. Y el One Hot usa `drop='first'` y `handle_unknown='ignore'`, como pide el agente A_09, para evitar multicolinealidad y errores con categorías nuevas.

## 5. Cómo leer `02_produccion_scoring.py`

| Bloque | Qué hace |
|---|---|
| Argumentos | Recibe dos parámetros, input (CSV de préstamos) y output (CSV de resultados) |
| Lectura | Lee el CSV y toma solo las 14 columnas predictoras |
| Tope de ingresos | Un ingreso mayor a 400.000 se puntúa como si fuera 400.000, para no distorsionar el escalado. Nadie se descarta |
| Carga y predicción | Carga los tres artefactos, que traen el preprocesamiento adentro, y predice. EAD y LGD se recortan a 0 y 1 |
| Pérdida esperada | PD por principal por EAD por LGD, como indica el manifiesto |
| Guardado | Escribe `id_cliente`, `pd_score`, `ead_score`, `lgd_score` y `perdida_esperada` |

El script no entrena nada. Trae los artefactos, les pasa todos los registros del CSV y da un score a cada cliente. Para volver a entrenar con datos nuevos se vuelve a ejecutar `01_reentrenamiento.py`.

## 6. Prueba de producción

Se ejecutó sobre `prestamos.csv` completo como prueba de funcionamiento.

| Revisión | Resultado |
|---|---|
| Filas de entrada y de salida | 200.000 y 200.000 |
| Valores nulos | Ninguno |
| PD media | 0,118, muy cerca de la tasa real de impago (0,1186) |
| EAD media | 0,675 |
| LGD media | 0,890 |
| Pérdida esperada media por préstamo | 1.180 |
| Pérdida esperada total | 236 millones |

Esta prueba incluye préstamos con los que se entrenó, así que confirma que el script funciona pero no sirve como evaluación. La evaluación válida es la de la reserva en la sección 3.

## 7. Decisiones tomadas

| Tema | Decisión |
|---|---|
| Reserva de validación | Se quita al principio del notebook y solo se usa para medir en `01_reentrenamiento.py` |
| Hiperparámetros | Se vuelven a buscar en cada reentrenamiento, con los de 07, 08 y 09 como punto de partida |
| Ingresos altos en producción | Se puntúan con un tope de 400.000 en vez de descartarse |
| Pérdida esperada | Se calcula en el script de scoring |
| Nota de cada modelo | La de la reserva, no la del artefacto final |

## 8. Lo que sigue

1. Construir la API con FastAPI que cargue los tres artefactos una sola vez.
2. Construir la app de Streamlit que consuma la API.
3. Desplegar en Render.
