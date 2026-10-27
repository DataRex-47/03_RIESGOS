# Informe de modelización

Proyecto Scoring de Riesgos, Caso 3. Elaborado por Andrés Miranda.

Estado: **PD, EAD y LGD terminados** (notebooks 07, 08 y 09). Sigue la preproducción (notebook 10).

## Qué mide cada modelo

| Modelo | Significa | Qué responde | Ejemplo |
|---|---|---|---|
| PD | Probabilidad de impago | ¿Qué tan probable es que el cliente deje de pagar? | 12 de cada 100 clientes |
| EAD | Exposición al impago | Si falla, ¿qué porcentaje del préstamo todavía debía? | 70 % del principal |
| LGD | Pérdida dado el impago | De lo que debía, ¿qué porcentaje no logra recuperar el banco? | 89 % de lo pendiente |

La pérdida esperada de un cliente es PD × principal × EAD × LGD.

**Tipo de modelo: adquisición.** Los tres se usan al decidir si se concede un préstamo, así que solo pueden usar variables que se conocen en ese momento. Por eso no entran el estado del préstamo, lo amortizado ni lo recuperado, y por eso el EAD y el LGD son difíciles de predecir: lo que pasa después del impago no se sabe al conceder.

---

## 1. Modelo PD: qué se hizo

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Datos | `df_tablon_pd.pkl`, objetivo `target_pd` | 139.602 préstamos, 29 predictoras |
| Reserva de validación | 70 % entrenamiento, 30 % validación, estratificada por el objetivo y con semilla fija (`random_state=42`) | 97.721 y 41.881 préstamos, con la misma proporción de impagos en ambos |
| Algoritmo | Regresión logística con `saga` y predictoras escaladas | Un solo algoritmo, sin preselección ni balanceo |
| Búsqueda | `GridSearchCV`, validación cruzada de 5, métrica AUC | 18 combinaciones, 90 entrenamientos |
| Elección | Lasso (`l1_ratio` 1) con C 0,05 | AUC de validación cruzada 0,696 |
| Evaluación | Una sola vez sobre la validación | **AUC 0,703** |

## 2. Qué se probó en el grid

| Parámetro | Valores | Qué se vio |
|---|---|---|
| `l1_ratio` | 0 (Ridge), 0,5 (ElasticNet), 1 (Lasso) | Los tres dan el mismo AUC |
| C | 0,01, 0,05, 0,1, 0,25, 0,75, 1 | Se aplana desde 0,1 |

| C | AUC medio de validación cruzada | Coeficientes en cero de 29 |
|---|---|---|
| 0,01 | 0,692 | no medido |
| 0,05 | 0,696 | 10 |
| 0,1 | 0,697 | 3 |
| 0,25 a 1 | 0,698 | 3 |

**Por qué C 0,05 y no el 1 que quedó primero:** la diferencia de AUC (0,002) es menor que la desviación entre folds (0,005), y con C 0,05 el modelo descarta 10 variables en lugar de 3. Es el modelo más simple que rinde igual, que es lo que busca Lasso.

## 3. Qué pesa en el modelo

| Variable | Efecto sobre el riesgo | Peso |
|---|---|---|
| `rating_scaled` | Sube | 3,655 |
| `tipo_interes_scaled` | Baja (signo inesperado) | 1,214 |
| `ingresos_scaled` | Baja | 1,146 |
| `num_lineas_credito_scaled` | Sube | 1,101 |
| `porc_uso_revolving_scaled` | Sube | 0,266 |
| `ingresos_verificados_Not Verified` | Baja | 0,268 |
| `imp_cuota_scaled`, `num_derogatorios` | Suben | 0,145 y 0,187 |
| 10 variables (dti, principal, finalidades como car o house, entre otras) | Sin efecto | 0 |

Los pesos son comparables porque las numéricas van escaladas de 0 a 1. Cada peso dice cuánto cambia el riesgo al pasar del mínimo al máximo de esa variable, con las demás fijas.

## 4. Dos pruebas hechas para entender el modelo

| Prueba | Resultado | Decisión |
|---|---|---|
| Correlación entre `rating_scaled` y `tipo_interes_scaled` | 0,953 | Explica el signo inesperado del interés: las dos miden casi lo mismo |
| Rating con one hot encoding en lugar de ordinal | AUC 0,699 contra 0,698 | Se deja el ordinal: rinde igual y es más simple |

Con el one hot, el riesgo sube de A a G, pero con saltos cada vez menores (A a B vale 0,86 y F a G 0,19). El ordinal supone saltos iguales y, aun así, no pierde AUC.

## 5. Evaluación sobre validación

| Medida | Valor | Lectura |
|---|---|---|
| AUC en validación | 0,703 | Coincide con la validación cruzada (0,696), no hay sobreajuste |
| Ganancia acumulada | Al revisar el 20 % de clientes con más riesgo se capturan cerca del 41 % de los impagos | Mejor que el azar, que capturaría el 20 % |
| Lift | Cerca de 2 en el 20 % más riesgoso, 1,65 en el 40 % | El modelo identifica el doble de impagos que el azar en ese grupo |

### Cómo leer la ganancia y el lift

La idea: ordenas a los clientes del que el modelo cree más riesgoso al menos riesgoso, y miras cuántos impagos reales encuentras al revisar solo a los primeros. Ejemplo con 1.000 clientes, de los cuales unos 119 caen en impago:

| Si revisas | Sin modelo (al azar) | Con el modelo | Lift |
|---|---|---|---|
| Los primeros 200 clientes (20 %) | Unos 24 impagos (20 %) | Unos 49 impagos (41 %) | 2,0 |
| Los primeros 400 clientes (40 %) | Unos 48 impagos (40 %) | Unos 79 impagos (66 %) | 1,65 |
| Los 1.000 clientes (100 %) | Los 119 (100 %) | Los 119 (100 %) | 1,0 |

| Gráfico | Qué muestra |
|---|---|
| Ganancia acumulada | % de clientes revisados (horizontal) contra % de los impagos que ya encontraste (vertical). La línea punteada es el azar: cuanto más arriba de ella, mejor |
| Lift | La misma información dividida: cuántas veces mejor que el azar. 1 es igual que el azar y 2 es el doble de impagos encontrados |

El lift termina siempre en 1 porque al revisar a todos ya no hay ventaja. La parte izquierda del lift sube y baja mucho porque son muy pocos clientes: conviene leerla desde el 10 % en adelante.

![Ganancia acumulada](grafico_gain_pd.png)

![Lift](grafico_lift_pd.png)

Un AUC de 0,70 es discreto: el modelo ordena mejor que el azar, pero no separa a los buenos de los malos con mucha fuerza. Es un resultado esperable con estas variables y un modelo lineal.

## 6. Modelo EAD: qué se hizo

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Datos | `df_tablon_ead.pkl`, objetivo `target_ead` (de 0 a 1) | 16.549 impagos, 29 predictoras (se quitan `target_pd` y el objetivo) |
| Reserva de validación | 70 % entrenamiento, 30 % validación | 4.965 préstamos para validar |
| Algoritmos probados | Ridge, Lasso y HistGradientBoosting, dentro de un Pipeline | 195 combinaciones, validación cruzada de 4 |
| Métrica del grid | Error absoluto medio (MAE) | Menor es mejor |
| Ganador | HistGradientBoosting: learning rate 0,05, profundidad 5, 200 iteraciones, l2 de 0,25, mínimo de 60 casos por hoja | |
| Corrección | Las predicciones se recortan a 0 y 1 | Un EAD no puede ser negativo ni pasar del 100 % |
| Evaluación | MAE sobre la validación | **0,158**: en promedio el modelo se equivoca en 16 puntos porcentuales |

Por qué el objetivo es un porcentaje y no un importe: así el modelo no puede predecir que un cliente debe más de lo que se prestó, y el MAE se lee directo en puntos porcentuales. Para obtener dinero se multiplica el resultado por el principal.

## 7. Cómo se comporta el EAD

![Real contra predicho](grafico_real_vs_pred_ead.png)

| Tramo del EAD real | Casos | Real medio | Predicho medio | Error medio |
|---|---|---|---|---|
| 0 a 20 % | 155 | 13 % | 63 % | 50 puntos |
| 20 a 40 % | 419 | 31 % | 64 % | 32 puntos |
| 40 a 60 % | 844 | 51 % | 66 % | 15 puntos |
| 60 a 80 % | 1.573 | 71 % | 68 % | 8 puntos |
| 80 a 100 % | 1.974 | 89 % | 73 % | 16 puntos |

**Qué se ve:** el modelo predice casi siempre entre el 60 % y el 80 %, sin importar el caso real. Las predicciones se agrupan alrededor de la media (cerca del 70 %) y no llegan a los extremos.

| Si el cliente debía de verdad | El modelo tiende a decir | Efecto |
|---|---|---|
| Poco (por ejemplo 13 %) | Mucho más (63 %) | Sobrestima: el error más grande del modelo, pero del lado prudente |
| Casi todo (por ejemplo 89 %) | Algo menos (73 %) | Subestima: el error es menor, pero es el lado que más pérdida puede traer |
| Lo típico (60 a 80 %) | Casi igual | Aquí acierta, porque coincide con la media |

**Por qué pasa:** con las variables conocidas al conceder el préstamo, el modelo no logra distinguir bien cuánto se había pagado cuando el cliente falló. Ante esa duda, predice un valor típico. Es un límite de los datos, y no un fallo de programación.

**Cómo explicarlo a negocio:** el modelo sirve para ordenar el riesgo de forma aproximada y es fiable para el caso típico, pero no para el monto exacto de un cliente en los extremos.

| Medida adicional | Valor |
|---|---|
| Error medio sobre el real medio | 22,7 % |
| Reporting | `reporting_regresion()`: nube de real contra predicho y densidades |

## 8. Modelo LGD: qué se hizo

| Paso | Qué se hizo | Resultado |
|---|---|---|
| Datos | `df_tablon_lgd.pkl`, objetivo `target_lgd` (de 0 a 1) | 16.549 impagos, 29 predictoras |
| Algoritmos probados | Los mismos que en EAD: Ridge, Lasso y HistGradientBoosting | 195 combinaciones, validación cruzada de 4, métrica MAE |
| Ganador | HistGradientBoosting: learning rate 0,01, profundidad 5, 50 iteraciones, l2 de 0 | La combinación más sencilla del grid |
| Evaluación | MAE sobre la validación | **0,088** (9,9 % sobre el real medio) |

## 9. Cómo se comporta el LGD

| Tramo del LGD real | Casos | Real medio | Predicho medio | Error medio |
|---|---|---|---|---|
| Hasta 0,854 (el 20 % con menor pérdida) | 993 | 0,706 | 0,888 | 0,182 |
| 0,854 a 0,866 | 993 | 0,860 | 0,888 | 0,028 |
| 0,866 a 0,942 | 993 | 0,900 | 0,889 | 0,017 |
| 0,942 a 1 | 1.986 | 0,995 | 0,889 | 0,106 |

**Qué se ve:** el modelo predice 0,888 o 0,889 para casi todos los clientes. No distingue entre unos y otros: es un modelo que responde casi lo mismo siempre.

### Pruebas para entender por qué

| Prueba | Resultado | Lectura |
|---|---|---|
| Comparar con un valor fijo para todos (la mediana del entrenamiento) | Error de 0,0876 frente a 0,0881 del modelo | El modelo no mejora esa referencia |
| Contar clientes por tramos del LGD real | El 87,6 % del entrenamiento está entre 0,8 y 1, y solo el 2,4 % pierde la mitad o menos | Hay muy pocos ejemplos de pérdidas bajas |
| Comparar real y predicho por rating | El LGD real medio es casi igual en todos los ratings (0,88 a 0,91) y el predicho también (0,887 a 0,889) | El modelo acierta el promedio de cada grupo, pero porque el LGD no cambia con el rating |
| Cambiar la métrica del grid al error cuadrático | MAE 0,0882 y predicciones entre 0,875 y 0,906 | No cambia nada: los datos no dan para más |
| Otros algoritmos (Random Forest, XGBoost) y menos hojas mínimas | No probados | Pendiente si se quiere confirmar que los datos no dan para más |

Por rating:

| Rating | Clientes | Real medio | Predicho medio |
|---|---|---|---|
| A | 282 | 0,906 | 0,889 |
| B | 945 | 0,894 | 0,889 |
| C | 1.565 | 0,893 | 0,889 |
| D | 1.144 | 0,885 | 0,889 |
| E | 671 | 0,891 | 0,887 |
| F | 278 | 0,881 | 0,887 |
| G | 80 | 0,897 | 0,887 |

**Conclusión:** con las variables disponibles al conceder el préstamo no se puede explicar cuánto se recupera después de un impago. En la práctica, el modelo de LGD equivale a asignar un valor fijo cercano al 89 %. Es un límite de los datos, no un fallo de programación, y es habitual: en la banca el LGD se trabaja a menudo con valores fijos por segmento.

**Hipótesis:** que en los datos el LGD se haya fijado por política de riesgo, con base en la experiencia y la prudencia del banco, o a partir de información de burós de crédito (según el consultor), en lugar de medirse caso a caso. Sería coherente con que casi no varíe entre clientes y con un banco de enfoque tradicional. No se puede comprobar con estos datos.

![Real contra predicho LGD](grafico_real_vs_pred_lgd.png)

## 10. Resumen de los tres modelos

| Modelo | Algoritmo | Resultado | Lectura |
|---|---|---|---|
| PD | Regresión logística con Lasso | AUC de validación 0,703 | Ordena el riesgo mejor que el azar, de forma discreta |
| EAD | HistGradientBoosting | MAE 0,158 | Acierta el caso típico y se agrupa hacia el 70 % en los extremos |
| LGD | HistGradientBoosting | MAE 0,088 | Equivale a un valor fijo cercano al 89 % |

La pérdida esperada de un cliente nuevo se calcula como PD × principal × EAD × LGD con los tres modelos. La parte con mayor poder predictivo es el PD.
