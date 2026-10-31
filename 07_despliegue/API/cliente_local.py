import json
from pathlib import Path

import requests


# ── Carga del payload de ejemplo ─────────
RUTA_PAYLOAD = Path(__file__).resolve().parent / "test_payload.json"
with open(RUTA_PAYLOAD, encoding="utf-8") as archivo:
    payload = json.load(archivo)


# ── Petición HTTP a la API local ─────────
respuesta = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload,
    timeout=30,
)

print("Código HTTP:", respuesta.status_code)
print("Respuesta:", respuesta.text)
