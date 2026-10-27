# Informe de EDA

## Alcance

El EDA se hizo sobre el conjunto de entrenamiento ya depurado en Calidad de Datos: 139.602 préstamos, separados en 7 variables categóricas (`cat_calidad.pkl`) y 16 numéricas (`num_calidad.pkl`). El 30 % de validación sigue reservado y no se usó. Al terminar se guardaron `cat_eda.pkl` y `num_eda.pkl` en `../02_datos/03_Entrenamiento/`.

En esta fase se renombró `antigüedad_empleo` como `antiguedad_empleo`, sin tilde ni ñ, para evitar problemas con el nombre de la columna.

## Variables categóricas

Se calcularon las frecuencias relativas de cada categoría y se graficaron con barras horizontales. Lo que se observó:

- `vivienda`: MORTGAGE 49,0 %, RENT 39,7 % y OWN 11,2 %. ANY, NONE y OTHER suman 70 registros en total, una presencia testimonial.
- `finalidad`: `debt_consolidation` es el 56,7 % y `credit_card` el 22,6 %. `wedding` (151 registros), `renewable_energy` (80) y `educational` (24) casi no tienen datos.
- `num_cuotas`: 71,4 % a 36 meses y 28,6 % a 60 meses.
- `rating`: B, C y A concentran el 77,3 % de los préstamos. G es el 0,5 % y F el 1,9 %.
- `ingresos_verificados`: Source Verified 39,1 %, Not Verified 33,2 % y Verified 27,8 %.
- `antiguedad_empleo`: 10+ years es la más frecuente (32,9 %). `Desconocido`, la categoría creada para los nulos, pesa el 6,5 %.
- `estado`: Fully Paid 47,6 %, Current 38,9 % y Charged Off 11,8 %. Las categorías Late, In Grace Period, Default y las de "Does not meet the credit policy" son muy pequeñas (Default solo tiene 3 registros).

**Decisiones para la transformación**

- En `vivienda`, juntar ANY, NONE y OTHER con la categoría mayoritaria.
- En `finalidad`, juntar `renewable_energy`, `educational` y `wedding` en `other`.
- Crear las variables objetivo a partir de `estado`.

## Variables numéricas

Se obtuvieron los estadísticos descriptivos (con la mediana junto a la media) y las densidades de las 16 variables. Lo que se observó:

- En `ingresos`, `principal`, `imp_cuota`, `imp_amortizado` y `tipo_interes` la media queda por encima de la mediana, señal de colas largas hacia la derecha. Ejemplo: `ingresos` tiene media 76.165 y mediana 65.000.
- `dti` tiene un máximo de 999 y un mínimo de -1, muy lejos de su mediana de 17,81.
- `num_derogatorios`: el 84,1 % de los préstamos tiene 0. Su máximo es 46.
- `num_hipotecas` tiene 2.825 préstamos con más de 6 hipotecas.
- `porc_uso_revolving` llegaba por encima de 100 %, un valor imposible.
- `num_meses_desde_ult_retraso`: el 51,4 % vale 0, mezcla de "sin retraso" (nulos imputados en Calidad) y retrasos reales.
- `id_cliente` entró en el análisis como si fuera una variable numérica, pero es un identificador.

**Importes del modelo LGD**

Se calcularon dos razones sobre el principal:

- `imp_amortizado / principal`: media 0,66, mediana 0,87 y percentil 75 en 1,0, es decir, la mayoría de los préstamos está total o casi totalmente amortizada.
- `imp_recuperado / principal`: media 0,009. Solo el 8,1 % de los préstamos tiene algo recuperado. El máximo es 1,22, o sea, 4 préstamos con más recuperado que el principal.

**Winsorización aplicada**

- `porc_uso_revolving` recortada a 100.
- `num_hipotecas` recortada a 7.
- `dti` recortada entre 0 y 100. El recorte primero quedó en una columna nueva llamada `DTI`, y después se corrigió para que `dti` quedara recortada y desapareciera `DTI`.

**Decisiones para la transformación**

- Convertir `num_derogatorios` en variable binaria.
- Tratar `dti` y `porc_uso_revolving` con los límites indicados; `num_hipotecas` con el suyo.

## Cruce entre estado e importe recuperado

Como el modelo LGD depende de `imp_recuperado`, se promedió esta variable por `estado`:

- Charged Off: 1.237,57 de recuperado promedio. De los 16.490 préstamos, 11.319 (68,6 %) tienen algo recuperado.
- Does not meet the credit policy. Status:Charged Off: 440,34 promedio, con los 56 préstamos con recuperación.
- Todos los demás estados, incluido Default (3 préstamos), tienen 0.

La conclusión es que solo hay recuperación en los préstamos que se dieron por perdidos. Esto confirma que EAD y LGD deben entrenarse únicamente con los préstamos en default. Con los tres estados de impago (Charged Off, Does not meet the credit policy. Status:Charged Off y Default) hay 16.549 préstamos, el 11,9 % del conjunto. El 38,9 % está en Current, así que su desenlace final aún no se conoce y la tasa de impago real será algo mayor.

## Resultado

- `cat_eda.pkl`: 139.602 filas y 7 columnas, sin nulos.
- `num_eda.pkl`: 139.602 filas y 15 columnas, sin nulos.

## Puntos abiertos y cómo siguen

- **Límite de `num_hipotecas`.** La conclusión habla de valores superiores a 6, pero el código recorta en 7. Esta variable no pasó a los tablones de la transformación.
- **`ingresos` igual a 0.** Siguen los 100 registros que ya se señalaron en Calidad, con `dti` en 0. No se resolvieron.
- **`num_meses_desde_ult_retraso`.** Resuelto en la transformación: se descartó la variable (ver `../Transformacion/informe_transformacion.md`).
- **`id_cliente`.** Resuelto en la transformación: queda como índice de los tablones y no como variable predictora.
- **Estados no resueltos.** El 38,9 % de los préstamos está en Current y el 1,5 % en Late o período de gracia, por lo que sus resultados finales aún no se conocen.
