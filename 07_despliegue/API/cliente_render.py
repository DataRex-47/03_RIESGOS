import json
from pathlib import Path

import requests

# ── Carga del payload de ejemplo ─────────
CARPETA = Path(__file__).resolve().parent
RUTA_PAYLOAD = CARPETA / "test_payload.json"
URL_API = "https://zero3-riesgos.onrender.com/predict"

with open(RUTA_PAYLOAD, encoding="utf-8") as archivo:
    payload = json.load(archivo)

# ── Petición HTTP a Render ─────────
respuesta = requests.post(
    URL_API,
    json=payload,
    timeout=120,
)

print("Código HTTP:", respuesta.status_code)
print("Respuesta:", respuesta.text)
