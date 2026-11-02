import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# ── Clave de acceso (opcional) ─────────
# La clave viaja en la cabecera X-API-Key, como una pulsera de concierto:
# quien no la muestra no pasa. Vive en una variable de entorno, nunca en el código.
#   - Sin API_KEY configurada la API queda abierta (demo pública y desarrollo local).
#   - Con API_KEY configurada, todas las rutas protegidas piden la clave.
cabecera = APIKeyHeader(name="X-API-Key", auto_error=False)


def verifica_clave(clave: str = Security(cabecera)):
    esperada = os.environ.get("API_KEY")

    # Sin clave configurada no hay nada que comprobar
    if not esperada:
        return

    # Si no llega clave, o no coincide, se responde 401 (no autorizado).
    # compare_digest compara sin revelar cuántas letras acertó.
    if clave is None or not secrets.compare_digest(clave.encode(), esperada.encode()):
        raise HTTPException(status_code=401, detail="Clave inválida o ausente")
