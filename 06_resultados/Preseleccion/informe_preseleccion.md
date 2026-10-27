# Informe de preselección y balanceo: conexión con la modelización

En este proyecto no se ejecutan los notebooks de preselección de variables ni de balanceo de clases. Este informe deja escritas las decisiones y lo que debe saber la modelización.

## 1. Decisiones

| Tema | Decisión | Por qué |
|---|---|---|
| Preselección de variables | No se hace | Hay 29 predictoras, pocas, y los modelos eligen variables por sí solos |
| Modelo PD | Regresión logística con regularización LASSO (L1) | L1 lleva a cero los coeficientes de las variables que no aportan, así que hace una preselección dentro del modelo |
| Modelo EAD | LightGBM | Elige variables al partir los árboles |
| Modelo LGD | LightGBM | Igual que EAD |
| Balanceo de clases | No se hace | Hay un 11,9 % de impagos y el modelo PD se mide con AUC |

Cómo funciona LASSO: si entran 50 variables y el modelo lleva 20 coeficientes a cero, en la práctica modela con 30. Para que funcione bien, las variables deben tener la misma escala, y por eso en la transformación se escaló todo con Min Max. Otra forma de preseleccionar es RFE con L1, que descarta variables una a una.

## 2. Variables que se pueden usar

Un modelo de adquisición solo puede usar variables que existan cuando se concede el préstamo. Si no, funciona en el notebook y falla en producción, porque la aplicación no puede pedirle ese dato a un cliente nuevo.

| Variables | ¿Se conocen al conceder? | Uso |
|---|---|---|
| `principal`, `tipo_interes`, `imp_cuota`, `num_cuotas`, `finalidad` | Sí | Predictoras |
| `rating` | Sí, nota de riesgo según agencias | Predictora |
| `ingresos`, `ingresos_verificados`, `vivienda`, `antiguedad_empleo` | Sí | Predictoras |
| `dti`, `num_lineas_credito`, `porc_uso_revolving`, `num_derogatorios` | Sí, historial crediticio al pedir el préstamo | Predictoras |
| `estado`, `imp_amortizado`, `imp_recuperado`, `pendiente` | No, ocurren después de conceder | Solo para crear las variables objetivo |

Con esto, las 29 predictoras de los tablones cumplen la regla.

## 3. Dos tipos de modelos de riesgo

| | Adquisición | Comportamiento |
|---|---|---|
| Momento | Cuando el cliente solicita el producto | Durante la vida del cliente |
| Información | Poca, solo la de ese instante | Cada mes más: pagos, uso, consumos |
| Qué decide | Conceder o no, y a qué precio | Cómo cambia el riesgo y qué hacer con la cartera |
| Ejemplo en banca | Aprobar una tarjeta a un cliente nuevo | Subir o bajar el límite según cómo la usa |
| Ejemplo en telecomunicaciones | Dar un plan con celular financiado | Detectar al cliente que paga tarde o se va |
| Ejemplo en seguros | Fijar la prima a un cliente nuevo | Renovar con descuento o recargo según los siniestros |

Este caso es de **adquisición**, aunque use LightGBM. La comparación justa de un modelo así es contra no tener modelo (la política actual), no contra uno de comportamiento o de marketing, que cuentan con años de historial. Más detalle en el vault: `Riesgo_y_Banca/Modelos_Adquisicion_vs_Comportamiento.md`.

## 4. Qué recibe la modelización

| Modelo | Tablón | Filas | Objetivo | Quitar de las predictoras |
|---|---|---|---|---|
| PD | `df_tablon_pd.pkl` | 139.602 | `target_pd` | Solo el objetivo |
| EAD | `df_tablon_ead.pkl` | 16.549 | `target_ead` | `target_pd` y el objetivo |
| LGD | `df_tablon_lgd.pkl` | 16.549 | `target_lgd` | `target_pd` y el objetivo |
