# Informe de la fase: app Streamlit de Bancandrex

## 1. El flujo completo

| Paso | Archivo | Qué hace | Qué produce |
|---|---|---|---|
| 1 | `app.py` (`carga_clientes`) | Al abrir, pide `GET /clientes` y guarda la lista | Los ids del selector |
| 2 | `app.py` (`dibuja_formulario`) | El analista elige cliente, monto, tasa, plazo, finalidad y vivienda | Una solicitud con seis campos |
| 3 | `app.py` (`despierta_api`, `llama_api`) | Despierta la API con `/health` y envía `POST /predict` con `[solicitud]` | La respuesta de la API |
| 4 | `API/main.py`, `schemas.py`, `scoring.py` | Valida, busca el historial, calcula la cuota y puntúa | PD, EAD, LGD y pérdida esperada |
| 5 | `app.py` (`dibuja_veredicto`) | Colorea el nivel de riesgo según la PD y dibuja las medidas | Banner, medidor, barras y cascada |
| 6 | `app.py` (`dibuja_comparador`) | Envía dos registros que solo cambian en el plazo | Dos tarjetas, 36 y 60 meses |

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

| Elemento de la app | Qué envía a la API o qué muestra | De dónde sale el dato |
|---|---|---|
| Cliente | Envía `id_cliente` | Lista de `GET /clientes` |
| Monto del préstamo | Envía `principal`, de 500 a 40,000 | El analista |
| Tasa de interés anual | Envía `tipo_interes`, de 5 a 31 | El analista |
| Plazo | Envía `num_cuotas`, 36 o 60 | El analista |
| Para qué es el préstamo | Envía `finalidad` | El analista, 14 opciones del contrato |
| Situación de vivienda | Envía `vivienda` | El analista, 4 opciones |
| Banner de riesgo | Nivel bajo, medio o alto, y qué hacer | `pd_score` contra los cortes 8 % y 15 % |
| Pérdida esperada | Dinero, porcentaje y valor por cada 100 prestados | `perdida_esperada` de la API y el monto escrito |
| Medidor de probabilidad de impago | Aguja con tres zonas | `pd_score` |
| Barra EAD | Porcentaje y dinero que quedaría por pagar | `ead_score` y el monto |
| Barra LGD | Porcentaje no recuperado y dinero perdido | `lgd_score`, `ead_score` y el monto |
| Cascada | Monto, exposición, pérdida si no paga, pérdida esperada | Monto y los tres puntajes; el último valor es el de la API |
| Aviso de datos cambiados | Pide volver a evaluar | El formulario actual contra la solicitud evaluada |
| Comparar plazos | Dos tarjetas con PD y pérdida esperada | Un solo POST con dos registros |

## 4. Qué pasa si falla

| Situación | Qué ve el usuario |
|---|---|
| API dormida | Mensaje de espera y barra de carga, hasta dos minutos |
| API apagada | No hay conexión con el servicio. Revisa que la API esté encendida |
| Tarda demasiado | El servicio tardó demasiado en responder. Inténtalo de nuevo |
| Error 404, cliente sin historial | El mensaje de la API |
| Error 422, dato fuera de rango | Algún dato está fuera de rango. Revisa el monto, la tasa o el plazo |
| Error 500 | No pudimos calcular el riesgo. Inténtalo de nuevo en un momento |

## 5. Qué se validó y qué no

| Prueba | Resultado |
|---|---|
| `GET /health` y `GET /clientes` | 200, y 50 ids |
| `POST /predict` con `test_payload.json` | 200; pérdida esperada 138.27 en la app, igual a monto × PD × EAD × LGD |
| App: cargar, evaluar y comparar (prueba automática de Streamlit) | Sin excepciones; 36 meses 138.27 y 60 meses 202.75 |
| Plazo de 60 meses | La app envía `num_cuotas` 60 y la API responde 200 |
| App en navegador Chromium, con capturas | Se ve y funciona, incluido el aviso de datos cambiados |
| API apagada | Muestra el mensaje de conexión, sin romperse |
| Variable de entorno `API_BASE_URL` | Probada: cambia la URL sin tocar el código |
| Códigos 404 y 422 | Probados directo contra la API; desde la app no se pueden provocar porque los controles ya limitan los valores |
| Sensibilidad a la tasa | Con el cliente 4155468 a 36 meses, subir la tasa de 6 % a 30 % baja la PD de 5.3 % a 0.7 %. Es lo que devuelve el modelo, no la app |
| Entorno de la prueba | Linux con Python 3.12.3, no tu Windows ni tu `.venv` |
| Render | Sin validar |

## 6. Cómo probarla

```
Terminal 1:  cd 07_despliegue\API   y luego   uv run uvicorn main:app --reload
Terminal 2:  cd 07_despliegue\app   y luego   uv run streamlit run app.py
```

Debe abrirse el navegador con el logo de Bancandrex y el selector con 50 clientes. Con los valores por defecto y el cliente 4155468, pulsar Evaluar solicitud muestra riesgo bajo y una pérdida esperada de unos 138 dólares.

## 7. Pendientes

1. Push de `API/main.py` y de `app/`, y crear el servicio de la app en Render con las dos variables de entorno.
2. Decidir qué hacer con la tasa: el modelo baja la PD cuando sube la tasa, porque aprendió la tasa junto con el rating.
3. Confirmar los cortes 8 % y 15 % con tu criterio.
