# Informe de la fase: app Streamlit de Bancandrex

Actualizado el 31/10/2026: la tasa deja de ser un control y pasa a ser un resultado, y la pantalla muestra la ficha del cliente.

## 1. El flujo completo

| Paso | Archivo | Qué hace | Qué produce |
|---|---|---|---|
| 1 | `app.py` (`carga_clientes`) | Al abrir, pide `GET /clientes` y guarda la lista | Los ids del selector |
| 2 | `app.py` (`carga_ficha`) | Pide `GET /clientes/{id}` al elegir un cliente | La ficha: lo que el banco ya sabe de él |
| 3 | `app.py` (`dibuja_formulario`) | El analista elige cliente, monto, plazo, finalidad y vivienda | Una solicitud con cinco campos |
| 4 | `app.py` (`despierta_api`, `llama_api`) | Despierta la API con `/health` y envía `POST /predict` con `[solicitud]` | La respuesta de la API |
| 5 | `API/main.py`, `schemas.py`, `scoring.py` | Valida, busca el historial, puntúa y calcula la tasa sugerida | PD, EAD, LGD, pérdida esperada y tasa sugerida |
| 6 | `app.py` (`dibuja_veredicto`) | Colorea el nivel de riesgo según la PD y dibuja las medidas | Banner, medidor, barras y cascada |
| 7 | `app.py` (`dibuja_comparador`) | Envía dos registros que solo cambian en el plazo | Dos tarjetas, 36 y 60 meses |

## 2. Cómo leer cada archivo

| Archivo | Qué es, en cristiano | Para qué sirve |
|---|---|---|
| `app.py` | El programa de la pantalla, en bloques por función | Dibuja la app y llama a la API |
| `assets/logo_bancandrex.png` | El logo, hecho para este proyecto | Cabecera |
| `assets/favicon_bancandrex.png` | El icono pequeño | Pestaña del navegador |
| `.streamlit/config.toml` | Los colores base de Streamlit | Tema claro y sin barra de herramientas |
| `idea/design_spec.json` | El diseño acordado | Fuente de lo que se construyó |
| `requirements.txt` | Lista de paquetes con versión exacta | Instalar en Render |
| `README.md` | Cómo ejecutarla y desplegarla | Guía |

## 3. La pantalla

La pregunta que responde la app es: **¿qué pasa si le prestamos esto a este cliente que ya conocemos?** El cliente existe en el banco y tiene historial. La app no da de alta clientes nuevos.

| Elemento de la app | Qué envía a la API o qué muestra | De dónde sale el dato |
|---|---|---|
| Cliente | Envía `id_cliente` | Lista de `GET /clientes` |
| Ficha del cliente | Rating, ingresos anuales, verificación de ingresos, DTI, antigüedad, líneas de crédito, uso rotativo y reportes negativos | `GET /clientes/{id}`. Son datos ficticios de demostración |
| Monto del préstamo | Envía `principal`, de 500 a 40.000 | El analista |
| Plazo | Envía `num_cuotas`, 36 o 60 | El analista |
| Para qué es el préstamo | Envía `finalidad` | El analista, 14 opciones del contrato |
| Situación de vivienda | Envía `vivienda` | El analista, 4 opciones. Se pide aquí porque el historial de la demo no la trae |
| Banner de riesgo | Nivel bajo, medio o alto, y qué hacer con la tasa: "se puede aprobar desde X %" | `pd_score` contra los cortes 8 % y 15 %, y `tasa_sugerida` |
| Pérdida esperada | Dinero, porcentaje y valor por cada 100 prestados | `perdida_esperada` de la API y el monto escrito |
| Tasa sugerida | Tasa anual mínima para cubrir la pérdida esperada | `tasa_sugerida` de la API |
| Leyenda del semáforo | Qué significa cada color | Los cortes definidos al inicio de `app.py` |
| Medidor de probabilidad de impago | Aguja con tres zonas | `pd_score` |
| Barra EAD | Porcentaje y dinero que quedaría por pagar | `ead_score` y el monto |
| Barra LGD | Porcentaje no recuperado y dinero perdido | `lgd_score`, `ead_score` y el monto |
| Cascada | Monto, exposición, pérdida si no paga, pérdida esperada | Monto y los tres puntajes; el último valor es el de la API |
| Aviso de datos cambiados | Pide volver a evaluar | El formulario actual contra la solicitud evaluada |
| Comparar plazos | Dos tarjetas con PD, pérdida esperada y tasa sugerida | Un solo POST con dos registros |
| Explicaciones desplegables | Cómo se calcula la pérdida esperada y la tasa sugerida | Texto en `app.py`, sin cifras fijas |

**Qué cambió con respecto a la primera versión.** Ya no hay control de tasa. Antes el analista escribía una tasa y el modelo la usaba para predecir el riesgo, lo cual es circular. Ahora la tasa es lo que el banco *debería cobrar* según el riesgo, y se ve en el banner. El semáforo se sigue calculando sobre la PD.

## 4. Qué pasa si falla

| Situación | Qué ve el usuario |
|---|---|
| API dormida | Mensaje de espera y barra de carga, hasta dos minutos |
| API apagada | No hay conexión con el servicio. Revisa que la API esté encendida |
| Tarda demasiado | El servicio tardó demasiado en responder. Inténtalo de nuevo |
| Error 404, cliente sin historial | El mensaje de la API |
| Error 422, dato fuera de rango | Algún dato está fuera de rango. Revisa el monto o el plazo |
| Error 500 | No pudimos calcular el riesgo. Inténtalo de nuevo en un momento |

## 5. Qué se validó y qué no

| Prueba | Resultado |
|---|---|
| `GET /health`, `GET /clientes` y `GET /clientes/{id}` | 200; 50 ids; ficha completa |
| `POST /predict` con `test_payload.json` | 200; pérdida esperada 297,41 y tasa sugerida 5,66 % para 15.000 a 36 meses |
| App: cargar, evaluar y comparar (prueba automática de Streamlit) | Sin excepciones; tasa sugerida visible en el banner y en las dos tarjetas de plazo |
| App en navegador Chromium, con capturas | Se ve y funciona, incluidas la ficha, la leyenda y el aviso de datos cambiados |
| API apagada | Muestra el mensaje de conexión, sin romperse |
| Variable de entorno `API_BASE_URL` | Cambia la URL sin tocar el código |
| Códigos 404 y 422 | Probados directo contra la API; desde la app no se pueden provocar porque los controles ya limitan los valores |
| Entorno de la prueba | Linux con Python 3.12, no tu Windows ni tu `.venv` |
| Render | Sin validar con los modelos nuevos: se valida después del push |

## 6. Cómo probarla

```
Terminal 1:  cd 07_despliegue\API   y luego   uv run uvicorn main:app --reload
Terminal 2:  cd 07_despliegue\app   y luego   uv run streamlit run app.py
```

Debe abrirse el navegador con el logo de Bancandrex, el selector con 50 clientes y, a su lado, la ficha del cliente elegido. Con el cliente de demostración 4155468, 10.000 a 36 meses, pulsar Evaluar solicitud muestra riesgo bajo, una pérdida esperada de unos 194 dólares y una tasa sugerida de 5,6 %.

## 7. Pendientes

1. Push de `07_despliegue/API` (modelos nuevos y rutas) y de `07_despliegue/app/`. El servicio de la app en Render no necesita cambios de variables.
2. Confirmar los cortes 8 % y 15 % con tu criterio. Con el modelo nuevo dejan 38 % de los préstamos en riesgo bajo, 37 % en medio y 25 % en alto.
3. Los supuestos de la tasa sugerida (fondeo 3,0 %, costos 0,7 %, margen 1,3 %) son ilustrativos y están documentados en el informe de la API.
