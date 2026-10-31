# Informe de modelización

Proyecto Scoring de Riesgos, Caso 3. Elaborado por Andrés Miranda.

Estado: **PD, EAD y LGD terminados** (notebooks 07, 08 y 09), reentrenados el 31/10/2026 sin la tasa de interés ni la cuota. Sigue la preproducción (notebook 10).

## Qué mide cada modelo

| Modelo | Significa | Qué responde | Ejemplo |
|---|---|---|---|
| PD | Probabilidad de impago | ¿Qué tan probable es que el cliente deje de pagar? | 12 de cada 100 clientes |
| EAD | Exposición al impago | Si falla, ¿qué porcentaje del préstamo todavía debía? | 70 % del principal |
| LGD | Pérdida dado el impago | De lo que debía, ¿qué porcentaje no logra recuperar el banco? | 89 % de lo pendiente |

La pérdida esperada de un cliente es PD × principal × EAD × LGD.

**Tipo de modelo: adquisición.** Los tres se usan al decidir si se concede un préstamo, así que solo pueden usar variables que se conocen en ese momento. Por eso no entran el estado del préstamo, lo amortizado ni lo recuperado, y por eso el EAD y el LGD son difíciles de predecir: lo que pasa después del impago no se sabe al conceder.

## Decisión del 31/10/2026: la tasa y la cuota salen de los modelos

En la primera versión, `tipo_interes` e `imp_cuota` entraban como predictoras. Se quitaron por tres razones:

| Motivo | Detalle |
|---|---|
| Es circular | La tasa la fija el banco al conceder, a partir del riesgo del cliente. Usarla para predecir ese mismo riesgo es usar la respuesta como pista |
| Casi repite el rating | La correlación entre `rating` y `tipo_interes` es 0,953, y el Lasso repartía el peso entre las dos con signos opuestos: con el mismo cliente, subir la tasa bajaba la PD |
| La cuota depende de la tasa | `imp_cuota` se calcula con el interés, así que arrastra lo mismo |

**Cuánto cuesta quitarlas.** Con validación cruzada sobre el 70 % de entrenamiento:

| Medida | Con tasa y cuota | Sin tasa ni cuota | Lectura |
|---|---|---|---|
| PD, AUC | 0,7002 | 0,6976 | Dentro del ruido de la validación cruzada (±0,005) |
| EAD, MAE | 0,1574 | 0,1623 | Medio punto porcentual peor |
| LGD, MAE | 0,0887 | 0,0887 | Igual |

La tasa deja de ser una entrada y pasa a ser una respuesta: la API devuelve una **tasa sugerida** calculada a partir de la pérdida esperada (ver el informe de la API).

---

## 1. Modelo PD: qué se hizo

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Datos | `df_tablon_pd.pkl`, objetivo `target_pd` | 139.602 préstamos, 27 predictoras |
| Reserva de validación | 70 % entrenamiento, 30 % validación, estratificada por el objetivo y con semilla fija (`random_state=42`) | 97.721 y 41.881 préstamos, con la misma proporción de impagos en ambos (11,9 %) |
| Algoritmo | Regresión logística con `saga` y predictoras escaladas | Un solo algoritmo, sin preselección ni balanceo |
| Búsqueda | `GridSearchCV`, validación cruzada de 5, métrica AUC | 18 combinaciones, 90 entrenamientos |
| Elección | Lasso (`l1_ratio` 1) con C 0,05 | AUC de validación cruzada 0,698 |
| Evaluación | Una sola vez sobre la validación | **AUC 0,694** |

## 2. Qué se probó en el grid

| Parámetro | Valores | Qué se vio |
|---|---|---|
| `l1_ratio` | 0 (Ridge), 0,5 (ElasticNet), 1 (Lasso) | Los tres dan el mismo AUC |
| C | 0,01, 0,05, 0,1, 0,25, 0,75, 1 | Se aplana desde 0,05 |

| C (Lasso) | AUC medio de validación cruzada | Coeficientes en cero de 27 |
|---|---|---|
| 0,01 | 0,695 | 17 |
| 0,05 | 0,698 | 10 |
| 0,1 | 0,699 | 3 |
| 0,25 a 1 | 0,699 | 2 o 3 |

**Por qué C 0,05 y no el 1 que quedó primero:** la diferencia de AUC (0,001) es menor que la desviación entre folds (0,003), y con C 0,05 el modelo descarta 10 variables en lugar de 2 o 3. Es el modelo más simple que rinde igual, que es lo que busca Lasso.

## 3. Qué pesa en el modelo

| Variable | Efecto sobre el riesgo | Peso |
|---|---|---|
| `rating_scaled` | Sube | 2,688 |
| `ingresos_scaled` | Baja | 1,094 |
| `num_lineas_credito_scaled` | Sube | 0,938 |
| `porc_uso_revolving_scaled` | Sube | 0,289 |
| `ingresos_verificados_Not Verified` | Baja | 0,253 |
| `num_derogatorios`, `vivienda_RENT` | Suben | 0,149 y 0,138 |
| `finalidad_other` | Baja | 0,122 |
| `ingresos_verificados_Verified` | Sube | 0,099 |
| `antiguedad_empleo_scaled`, `vivienda_MORTGAGE` | Bajan | 0,061 y 0,059 |
| `principal_scaled`, `finalidad_debt_consolidation` | Suben, muy poco | 0,039 y 0,036 |
| 10 variables (plazo, `vivienda_OWN`, finalidades como car, house o medical, entre otras) | Sin efecto | 0 |

Los pesos son comparables porque las numéricas van escaladas de 0 a 1. Cada peso dice cuánto cambia el riesgo al pasar del mínimo al máximo de esa variable, con las demás fijas.

**Lectura:** ahora el rating manda (2,7), y los ingresos y las líneas de crédito lo siguen. Desaparece el signo raro de la tasa. El plazo (36 o 60 meses) no pesa en la PD: los pesos de `num_cuotas` quedan en cero.

Quedó pendiente de la versión anterior una prueba (rating con one hot en lugar de ordinal). No se repitió: el ordinal rinde bien y es más simple.

## 4. Evaluación sobre validación

| Medida | Valor | Lectura |
|---|---|---|
| AUC en validación | 0,694 | Cercano a la validación cruzada (0,698), no hay sobreajuste |
| Ganancia acumulada | Al revisar el 10 %, 20 % y 40 % de clientes con más riesgo se capturan el 23 %, 40 % y 64 % de los impagos | Mejor que el azar, que capturaría el 10 %, 20 % y 40 % |
| Lift | 2,3 en el 10 % más riesgoso, 2,0 en el 20 % y 1,6 en el 40 % | El modelo identifica el doble de impagos que el azar en ese grupo |

### Cómo leer la ganancia y el lift

La idea: ordenas a los clientes del que el modelo cree más riesgoso al menos riesgoso, y miras cuántos impagos reales encuentras al revisar solo a los primeros. Ejemplo con 1.000 clientes, de los cuales unos 119 caen en impago:

| Si revisas | Sin modelo (al azar) | Con el modelo | Lift |
|---|---|---|---|
| Los primeros 200 clientes (20 %) | Unos 24 impagos (20 %) | Unos 47 impagos (40 %) | 2,0 |
| Los primeros 400 clientes (40 %) | Unos 48 impagos (40 %) | Unos 77 impagos (64 %) | 1,6 |
| Los 1.000 clientes (100 %) | Los 119 (100 %) | Los 119 (100 %) | 1,0 |

| Gráfico | Qué muestra |
|---|---|
| Ganancia acumulada | % de clientes revisados (horizontal) contra % de los impagos que ya encontraste (vertical). La línea punteada es el azar: cuanto más arriba de ella, mejor |
| Lift | La misma información dividida: cuántas veces mejor que el azar. 1 es igual que el azar y 2 es el doble de impagos encontrados |

El lift termina siempre en 1 porque al revisar a todos ya no hay ventaja. La parte izquierda del lift sube y baja mucho porque son muy pocos clientes: conviene leerla desde el 10 % en adelante.

![Ganancia acumulada](grafico_gain_pd.png)

![Lift](grafico_lift_pd.png)

Un AUC de 0,69 es discreto: el modelo ordena mejor que el azar, pero no separa a los buenos de los malos con mucha fuerza. Es un resultado esperable con estas variables y un modelo lineal. Quitar la tasa lo bajó apenas de 0,703 a 0,694 en validación.

## 5. Modelo EAD: qué se hizo

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Datos | `df_tablon_ead.pkl`, objetivo `target_ead` (de 0 a 1) | 16.549 impagos, 27 predictoras (se quitan `target_pd` y el objetivo) |
| Reserva de validación | 70 % entrenamiento, 30 % validación | 4.965 préstamos para validar |
| Algoritmos probados | Ridge, Lasso y HistGradientBoosting, dentro de un Pipeline | 195 combinaciones, validación cruzada de 4 |
| Métrica del grid | Error absoluto medio (MAE) | Menor es mejor |
| Ganador | Ridge con alpha 0,1 | Sin la tasa, el grid prefiere un modelo lineal; con HistGradientBoosting el error es parecido |
| Corrección | Las predicciones se recortan a 0 y 1 | Un EAD no puede ser negativo ni pasar del 100 % |
| Evaluación | MAE sobre la validación | **0,162**: en promedio el modelo se equivoca en 16 puntos porcentuales |

Por qué el objetivo es un porcentaje y no un importe: así el modelo no puede predecir que un cliente debe más de lo que se prestó, y el MAE se lee directo en puntos porcentuales. Para obtener dinero se multiplica el resultado por el principal.

## 6. Cómo se comporta el EAD

![Real contra predicho](grafico_real_vs_pred_ead.png)

| Tramo del EAD real | Casos | Real medio | Predicho medio | Error medio |
|---|---|---|---|---|
| 0 a 20 % | 188 | 13 % | 64 % | 51 puntos |
| 20 a 40 % | 391 | 31 % | 64 % | 33 puntos |
| 40 a 60 % | 813 | 51 % | 67 % | 16 puntos |
| 60 a 80 % | 1.582 | 71 % | 69 % | 8 puntos |
| 80 a 100 % | 1.991 | 89 % | 73 % | 17 puntos |

**Qué se ve:** el modelo predice casi siempre entre el 60 % y el 75 %, sin importar el caso real. Las predicciones se agrupan alrededor de la media (cerca del 70 %) y no llegan a los extremos.

| Si el cliente debía de verdad | El modelo tiende a decir | Efecto |
|---|---|---|
| Poco (por ejemplo 13 %) | Mucho más (64 %) | Sobrestima: el error más grande del modelo, pero del lado prudente |
| Casi todo (por ejemplo 89 %) | Algo menos (73 %) | Subestima: el error es menor, pero es el lado que más pérdida puede traer |
| Lo típico (60 a 80 %) | Casi igual | Aquí acierta, porque coincide con la media |

**Por qué pasa:** con las variables conocidas al conceder el préstamo, el modelo no logra distinguir bien cuánto se había pagado cuando el cliente falló. Ante esa duda, predice un valor típico. Es un límite de los datos, y no un fallo de programación.

**Cómo explicarlo a negocio:** el modelo sirve para ordenar el riesgo de forma aproximada y es fiable para el caso típico, pero no para el monto exacto de un cliente en los extremos.

| Medida adicional | Valor |
|---|---|
| Error medio sobre el real medio | 23,3 % |
| Reporting | `reporting_regresion()`: nube de real contra predicho y densidades |

## 7. Modelo LGD: qué se hizo

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Datos | `df_tablon_lgd.pkl`, objetivo `target_lgd` (de 0 a 1) | 16.549 impagos, 27 predictoras |
| Algoritmos probados | Los mismos que en EAD: Ridge, Lasso y HistGradientBoosting | 195 combinaciones, validación cruzada de 4, métrica MAE |
| Ganador | Lasso con alpha 0,0001 | Casi sin penalización: el LGD casi no se puede predecir y da igual el algoritmo |
| Evaluación | MAE sobre la validación | **0,088** (9,9 % sobre el real medio) |

## 8. Cómo se comporta el LGD

| Tramo del LGD real | Casos | Real medio | Predicho medio | Error medio |
|---|---|---|---|---|
| Hasta 0,854 (el 20 % con menor pérdida) | 993 | 0,705 | 0,887 | 0,182 |
| 0,854 a 0,867 | 993 | 0,860 | 0,889 | 0,029 |
| 0,867 a 0,942 | 993 | 0,901 | 0,888 | 0,020 |
| 0,942 a 1 | 1.986 | 0,996 | 0,890 | 0,106 |

**Qué se ve:** el modelo predice entre 0,887 y 0,890 para casi todos los clientes. No distingue entre unos y otros: es un modelo que responde casi lo mismo siempre.

### Pruebas para entender por qué

| Prueba | Resultado | Lectura |
|---|---|---|
| Comparar con un valor fijo para todos (la mediana del entrenamiento) | Error de 0,0879 frente a 0,0885 del modelo | El modelo no mejora esa referencia |
| Contar clientes por tramos del LGD real | El 87,6 % del entrenamiento está entre 0,8 y 1, y solo el 2,4 % pierde la mitad o menos | Hay muy pocos ejemplos de pérdidas bajas |
| Comparar real y predicho por rating | El LGD real medio es casi igual en todos los ratings (0,885 a 0,906) y el predicho también (0,880 a 0,894) | El modelo acierta el promedio de cada grupo, pero porque el LGD no cambia con el rating |
| Otros algoritmos (Random Forest, XGBoost) y menos hojas mínimas | No probados | Pendiente si se quiere confirmar que los datos no dan para más |

Por rating:

| Rating | Clientes | Real medio | Predicho medio |
|---|---|---|---|
| A | 263 | 0,906 | 0,894 |
| B | 968 | 0,896 | 0,892 |
| C | 1.597 | 0,888 | 0,890 |
| D | 1.096 | 0,894 | 0,887 |
| E | 686 | 0,889 | 0,885 |
| F | 269 | 0,885 | 0,883 |
| G | 86 | 0,893 | 0,880 |

**Conclusión:** con las variables disponibles al conceder el préstamo no se puede explicar cuánto se recupera después de un impago. En la práctica, el modelo de LGD equivale a asignar un valor fijo cercano al 89 %. Es un límite de los datos, no un fallo de programación, y es habitual: en la banca el LGD se trabaja a menudo con valores fijos por segmento.

**Hipótesis:** que en los datos el LGD se haya fijado por política de riesgo, con base en la experiencia y la prudencia del banco, o a partir de información de burós de crédito (según el consultor), en lugar de medirse caso a caso. Sería coherente con que casi no varíe entre clientes y con un banco de enfoque tradicional. No se puede comprobar con estos datos.

![Real contra predicho LGD](grafico_real_vs_pred_lgd.png)

## 9. Resumen de los tres modelos

| Modelo | Algoritmo | Resultado | Lectura |
|---|---|---|---|
| PD | Regresión logística con Lasso | AUC de validación 0,694 | Ordena el riesgo mejor que el azar, de forma discreta |
| EAD | Ridge | MAE 0,162 | Acierta el caso típico y se agrupa hacia el 70 % en los extremos |
| LGD | Lasso | MAE 0,088 | Equivale a un valor fijo cercano al 89 % |

La pérdida esperada de un cliente nuevo se calcula como PD × principal × EAD × LGD con los tres modelos. La parte con mayor poder predictivo es el PD: el EAD y el LGD se parecen a valores casi constantes (alrededor del 66 % y del 89 %). Esto se dice con claridad en el README porque limita lo que el sistema puede prometer.

Para producción, el script `01_reentrenamiento.py` repite la búsqueda con otro método (búsqueda aleatoria y reserva por clientes) y puede escoger otros hiperparámetros o algoritmos; las diferencias están dentro del ruido y se detallan en el informe de preproducción.
