# Informe de calidad de datos

## Alcance

La fase de calidad se realizó sobre el conjunto de entrenamiento: 140.000 préstamos, el 70 % de los 200.000 de `prestamos.csv`. El 30 % restante (60.000) está reservado para validación en `prestamos_val.pkl`, se mantiene separado y no se usa en esta fase.

## Variables principales

- `porc_tarjetas_75p`: porcentaje de tarjetas cuyo saldo supera el 75 % de su límite. Por ejemplo, si una tarjeta tiene un saldo de 800 sobre un límite de 1.000 y otra de 500 sobre 1.000, una de las dos supera el 75 %: el valor es 50 %.
- `porc_uso_revolving`: porcentaje del crédito revolving disponible que está utilizado. El crédito revolving permite mantener un saldo pendiente y volver a disponer del crédito; si no se paga el saldo completo, pueden generarse intereses y la deuda puede prolongarse. Un porcentaje alto indica que se está usando gran parte del crédito disponible y puede ser una señal de riesgo.
- `dti` (*debt to income*): relación entre la deuda total del cliente y sus ingresos mensuales, sin contar las hipotecas. Un valor alto indica que una parte importante de los ingresos ya está comprometida en pagos de deuda.
- `num_derogatorios`: número de incidencias negativas registradas en el historial crediticio, como registros públicos desfavorables. Es un recuento y debe tratarse como entero.
- `finalidad`: motivo declarado para solicitar el préstamo. `debt_consolidation` indica que se busca agrupar varias deudas en un nuevo préstamo; si sus condiciones son mejores, puede simplificar los pagos o reducir el coste financiero.
- `principal`: importe inicial prestado, antes de intereses y otros cargos.
- `tipo_interes`: tasa de interés aplicada al préstamo.
- `num_cuotas`: número de pagos previstos para devolver el préstamo.
- `imp_cuota`: importe de cada cuota periódica.
- `imp_amortizado`: importe del principal que ya se ha devuelto.
- `estado`: situación del préstamo en el momento en que se observó el registro.
- `imp_recuperado`: importe recuperado por el prestamista después de que el préstamo entrara en impago.

## Diagnóstico

- No hay registros duplicados.
- `id_prestamo` está vacía (0 valores en todo el conjunto), por eso se elimina.
- Nulos en las variables categóricas: `descripcion` 132.208 (94 %), `empleo` 10.310 y `antigüedad_empleo` 9.052.
- Nulos en las variables numéricas: `num_meses_desde_ult_retraso` 71.743 (51 %), `porc_tarjetas_75p` 4.609, `num_hipotecas` 3.073, `porc_uso_revolving` 109, `dti` 102 y cantidades mínimas (8 o menos) en `num_cancelaciones_12meses`, `num_lineas_credito` y `num_derogatorios`.

## Tratamientos y decisiones

- Se eliminaron `id_prestamo`, por estar vacía, y las columnas `empleo` y `descripcion`, que parecen texto libre (ver conclusiones).
- Los nulos de `antigüedad_empleo` se imputaron con la categoría `Desconocido`.
- Los nulos numéricos se imputaron con cero. El criterio fue que un nulo significa ausencia: sin retrasos en `num_meses_desde_ult_retraso`, sin tarjetas en `porc_tarjetas_75p` y sin hipotecas en `num_hipotecas`. Después, los recuentos (`num_hipotecas`, `num_lineas_credito`, `num_cancelaciones_12meses`, `num_derogatorios` y `num_meses_desde_ult_retraso`) se convirtieron a entero.
- Atípicos: se revisaron `ingresos`, `principal`, `imp_cuota` e `imp_recuperado` con el método de las desviaciones típicas (3,7). El análisis en detalle se centró en `ingresos`, con un máximo de 10.999.200, muy lejos del resto; por eso se eliminaron los registros con ingresos superiores a 400.000 $. Fueron 398 registros (0,28 %), eliminados a la vez de los conjuntos categórico y numérico para que sus filas sigan alineadas.
- Resultado: 139.602 préstamos, guardados en `../02_datos/03_Entrenamiento/train_calidad.pkl`, `cat_calidad.pkl` y `num_calidad.pkl`.

## Puntos que quedaron abiertos para el EDA y cómo siguen

- `num_meses_desde_ult_retraso`: imputar con cero los nulos equivale a decir "hace 0 meses del último retraso", que se lee como un retraso reciente. Aquí el cero significa "nunca tuvo retraso". El EDA mostró que el 51,4 % de la columna vale 0, y en la transformación se descartó la variable.
- `ingresos` igual a 0: 100 registros, que además tienen `dti` en 0. Siguen sin tratarse.
- `dti`: 1 valor negativo y 163 superiores a 100, que no tienen sentido de negocio. Resuelto: en el EDA se recortó entre 0 y 100.
- `porc_uso_revolving`: 402 valores superiores a 100 %, imposibles. Resuelto en el EDA: se recortó a 100.
- `dti` y `porc_uso_revolving` también tenían nulos (102 y 109) que se imputaron con cero; son pocos, pero un `dti` de 0 equivale a decir que no tiene deudas. Sin cambios.

## Conclusiones y trabajo futuro

La alta variabilidad de `empleo` y `descripcion` sugiere que se registraron como texto libre, sin un catálogo suficientemente normalizado. Se excluyen en esta fase para evitar tratar esos textos como categorías inconsistentes.

Se recomienda abordar estos campos en un pequeño proyecto específico: analizar y normalizar los textos, definir categorías y evaluar si permiten crear variables útiles. La solución podría servir también para mejorar la captura y el tratamiento de futuros textos libres.