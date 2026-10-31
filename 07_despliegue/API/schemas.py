from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Valores que el modelo conoce ─────────
# Son los mismos valores del dataset. Si llega otro, la API lo rechaza (error 422),
# porque el modelo lo trataría en silencio como si fuera el valor de referencia.
VIVIENDA = Literal["RENT", "OWN", "MORTGAGE", "ANY", "NONE", "OTHER"]
FINALIDAD = Literal[
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house",
    "other", "wedding", "educational", "renewable_energy",
]


# ── Lo que envía la app: una solicitud de préstamo ─────────
class RegistroEntrada(BaseModel):
    # Si llega un campo que no está en esta lista, se rechaza (error 422)
    model_config = ConfigDict(extra="forbid")

    # Un campo sin valor por defecto es obligatorio
    id_cliente: int

    # Los rangos son los que el modelo vio al entrenar
    principal: float = Field(ge=500, le=40000)
    tipo_interes: float = Field(ge=5, le=31)  # porcentaje anual, por ejemplo 7.21

    num_cuotas: Literal[36, 60]
    finalidad: FINALIDAD
    vivienda: VIVIENDA


# ── Lo que devuelve la API por cada solicitud ─────────
class ScoringSalida(BaseModel):
    id_cliente: int
    pd_score: float
    ead_score: float
    lgd_score: float
    perdida_esperada: float
