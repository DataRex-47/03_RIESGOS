from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Datos que llegan desde la aplicación ─────────
class RegistroEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_cliente: int = Field(..., alias="id_cliente")
    principal: float = Field(..., gt=0, alias="principal")
    tipo_interes: float = Field(..., ge=0, alias="tipo_interes")
    num_cuotas: Literal[36, 60] = Field(..., alias="num_cuotas")
    finalidad: str = Field(..., alias="finalidad")
    vivienda: str = Field(..., alias="vivienda")


# ── Resultado completo del scoring ─────────
class ScoringSalida(BaseModel):
    id_cliente: int = Field(..., alias="id_cliente")
    pd_score: float = Field(..., alias="pd_score")
    ead_score: float = Field(..., alias="ead_score")
    lgd_score: float = Field(..., alias="lgd_score")
    perdida_esperada: float = Field(..., alias="perdida_esperada")
