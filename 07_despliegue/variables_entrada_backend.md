# Variables de entrada del backend

Actualizado el 31/10/2026: `tipo_interes` e `imp_cuota` ya no son entradas. La tasa es una salida (`tasa_sugerida`).

## Entrada desde la app (la solicitud)

- `id_cliente`: cliente que ya existe en el banco
- `principal`: entre 500 y 40.000
- `num_cuotas`: 36 o 60 (los únicos plazos que hay en los datos)
- `finalidad`
- `vivienda`: se pide en la solicitud porque el historial de la demo no la trae

Cualquier otro campo, por ejemplo `tipo_interes`, se rechaza con error 422.

## Entrada desde CSV: historial del cliente

- `ingresos`
- `ingresos_verificados`
- `rating`
- `dti`
- `antigüedad_empleo`
- `num_lineas_credito`
- `porc_uso_revolving`
- `num_derogatorios`

## Variables que ya no existen

- `tipo_interes` y `imp_cuota`: salieron de los modelos porque la tasa la fija el banco según el riesgo del cliente, y usarla para predecir ese riesgo es circular.

## Salida calculada

- `pd_score`, `ead_score`, `lgd_score` y `perdida_esperada` (PD × principal × EAD × LGD).
- `tasa_sugerida`: tasa anual mínima en %, igual a costo de fondeo (3,0) más costos operativos (0,7) más margen (1,3) más la pérdida esperada anual (`100 × perdida_esperada / principal / (num_cuotas / 12)`). Los tres primeros son supuestos ilustrativos y están en `scoring.py`.
- `num_cuotas` se recibe como `36` o `60` y se transforma al texto que espera el modelo (`36 months` o `60 months`).

## Flujo de la API

- `main.py` busca `id_cliente` en `API/demo_clientes.csv` y combina las ocho variables de historial con los cinco datos de la solicitud.
- El archivo de historial incluido en `API/` es una copia de `07_despliegue/demo_clientes.csv` para que los datos requeridos queden dentro de la raíz desplegable del servicio.
- `GET /clientes` entrega solo los ids y `GET /clientes/{id}` entrega la ficha (nunca se publica el CSV completo).
