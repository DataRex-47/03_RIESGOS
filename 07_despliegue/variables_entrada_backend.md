# Variables de entrada del backend

## Entrada desde la app

- `id_cliente`
- `principal`
- `tipo_interes`
- `num_cuotas`
- `finalidad`
- `vivienda`

## Entrada desde CSV: historial del cliente

- `ingresos`
- `ingresos_verificados`
- `rating`
- `dti`
- `antigüedad_empleo`
- `num_lineas_credito`
- `porc_uso_revolving`
- `num_derogatorios`

## Variable calculada internamente

- `imp_cuota`: cuota mensual calculada con el sistema francés. `tipo_interes` se recibe como porcentaje anual (por ejemplo, `7.21` significa 7,21 %) y se convierte a tasa mensual dividiendo entre 100 y entre 12. `num_cuotas` se recibe como `36` o `60` y se transforma al texto que espera el modelo (`36 months` o `60 months`).

## Flujo de la API

- `main.py` busca `id_cliente` en `api/demo_clientes.csv` y combina las ocho variables de historial con los seis datos de la solicitud.
- El archivo de historial incluido en `api/` es una copia de `07_despliegue/demo_clientes.csv` para que los datos requeridos queden dentro de la raíz desplegable del servicio.
