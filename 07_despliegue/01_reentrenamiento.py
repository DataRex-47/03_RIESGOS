from pathlib import Path

import cloudpickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

# ── Rutas y configuración ─────────
RAIZ = Path(__file__).resolve().parents[1]
CSV_PATH = RAIZ / "02_datos" / "01_Originales" / "prestamos.csv"
VALIDACION_PATH = RAIZ / "02_datos" / "02_Validacion" / "prestamos_val.pkl"
DIRECTORIO_ARTEFACTOS = RAIZ / "07_despliegue"
ID = "id_cliente"
ESTADOS_DEFAULT = [
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
    "Default",
]
COLUMNAS_CATEGORICAS = ["ingresos_verificados", "vivienda", "finalidad", "num_cuotas"]
COLUMNAS_ORDINALES = ["antigüedad_empleo", "rating"]
COLUMNAS_NUMERICAS = [
    "ingresos",
    "dti",
    "num_lineas_credito",
    "porc_uso_revolving",
    "principal",
    "tipo_interes",
    "imp_cuota",
]
COLUMNAS_BINARIAS = ["num_derogatorios"]
COLUMNAS_PREDICTORAS = [
    "ingresos_verificados",
    "vivienda",
    "finalidad",
    "num_cuotas",
    "antigüedad_empleo",
    "rating",
    "ingresos",
    "dti",
    "num_lineas_credito",
    "porc_uso_revolving",
    "principal",
    "tipo_interes",
    "imp_cuota",
    "num_derogatorios",
]
ORDEN_ANTIGUEDAD = [
    "Desconocido",
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]
ORDEN_RATING = ["A", "B", "C", "D", "E", "F", "G"]

# ── Funciones por fase del proyecto ─────────
# Calidad de datos: excluye préstamos con ingresos superiores a 400.000.
def prepara_datos(df: pd.DataFrame) -> pd.DataFrame:
    # Replica el filtro de calidad; los ingresos ausentes se trataban como cero en esa fase.
    ingresos = pd.to_numeric(df["ingresos"], errors="coerce").fillna(0)
    return df.loc[ingresos <= 400000].copy()


# Transformación de datos: agrupa categorías poco frecuentes según las decisiones del proyecto.
def unifica_categorias(X: pd.DataFrame) -> pd.DataFrame:
    # Se aplica antes del imputador para conservar los nombres de las columnas y usar pandas.
    X = X.copy()
    X["vivienda"] = X["vivienda"].replace({"ANY": "MORTGAGE", "NONE": "MORTGAGE", "OTHER": "MORTGAGE"})
    X["finalidad"] = X["finalidad"].replace(
        {"wedding": "other", "educational": "other", "renewable_energy": "other"}
    )
    return X


# EDA: aplica los límites definidos para dti y porcentaje de uso revolving.
def limita_numericas(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["dti"] = X["dti"].clip(lower=0, upper=100)
    X["porc_uso_revolving"] = X["porc_uso_revolving"].clip(lower=0, upper=100)
    return X


# Transformación de datos: deriva PD, EAD y LGD; solo se usan como objetivos.
def crea_objetivos(df: pd.DataFrame) -> pd.DataFrame:
    # Los objetivos solo se calculan para entrenar y medir; nunca entran como predictoras.
    target_pd = df["estado"].isin(ESTADOS_DEFAULT).astype("int8")
    principal = pd.to_numeric(df["principal"], errors="coerce").fillna(0)
    amortizado = pd.to_numeric(df["imp_amortizado"], errors="coerce").fillna(0)
    recuperado = pd.to_numeric(df["imp_recuperado"], errors="coerce").fillna(0)
    pendiente = principal - amortizado

    # Los denominadores cero producen valores nulos, que se convierten a cero como en la preparación.
    denominador_ead = principal.where(principal.ne(0))
    denominador_lgd = pendiente.where(pendiente.ne(0))
    target_ead = pendiente.div(denominador_ead).fillna(0).clip(lower=0, upper=1)
    target_lgd = (1 - recuperado.div(denominador_lgd)).fillna(0).clip(lower=0, upper=1)

    objetivos = pd.DataFrame(index=df.index)
    objetivos["target_pd"] = target_pd
    objetivos["target_ead"] = target_ead
    objetivos["target_lgd"] = target_lgd
    return objetivos


# Modelización: define el preprocesamiento que se ajusta dentro de cada pipeline.
def crea_preprocesador():
    # Los transformadores personalizados trabajan con DataFrames para que sus columnas sean legibles.
    # Cada grupo conserva el tratamiento definido en la fase de transformación del proyecto.
    transformador_categorico = make_pipeline(
        FunctionTransformer(unifica_categorias, validate=False),
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        ),
    )
    transformador_ordinal = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Desconocido"),
        OrdinalEncoder(
            categories=[ORDEN_ANTIGUEDAD, ORDEN_RATING],
            handle_unknown="use_encoded_value",
            unknown_value=12,
        ),
        MinMaxScaler(),
    )
    transformador_numerico = make_pipeline(
        FunctionTransformer(limita_numericas, validate=False),
        SimpleImputer(strategy="constant", fill_value=0),
        MinMaxScaler(),
    )
    transformador_binario = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        Binarizer(threshold=0),
    )
    return make_column_transformer(
        (transformador_categorico, COLUMNAS_CATEGORICAS),
        (transformador_ordinal, COLUMNAS_ORDINALES),
        (transformador_numerico, COLUMNAS_NUMERICAS),
        (transformador_binario, COLUMNAS_BINARIAS),
        remainder="drop",
    )


# Modelización: combina el preprocesamiento y el estimador candidato.
def crea_pipeline(nombre_modelo: str):
    preprocesador = crea_preprocesador()
    if nombre_modelo == "PD":
        modelo_base = LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=42,
        )
    else:
        modelo_base = HistGradientBoostingRegressor(
            min_samples_leaf=60,
            scoring="neg_mean_absolute_error",
            random_state=42,
        )
    return make_pipeline(preprocesador, modelo_base)


# Modelización: mide MAE con las predicciones EAD/LGD dentro del rango [0, 1].
def mae_acotado(y_real, y_predicho):
    predicciones = pd.Series(y_predicho).clip(lower=0, upper=1)
    return mean_absolute_error(y_real, predicciones)


# ── Espacios de búsqueda ─────────
# Los regresores se prueban en el mismo paso del pipeline junto con sus hiperparámetros.
PARAMETROS_PD = {
    "logisticregression__l1_ratio": [0, 0.5, 1],
    "logisticregression__C": [0.01, 0.05, 0.1, 0.25, 0.75, 1],
}
PARAMETROS_REGRESION = [
    {
        "histgradientboostingregressor": [Ridge()],
        "histgradientboostingregressor__alpha": [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
    },
    {
        "histgradientboostingregressor": [Lasso(max_iter=10000)],
        "histgradientboostingregressor__alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
    },
    {
        "histgradientboostingregressor": [
            HistGradientBoostingRegressor(
                min_samples_leaf=60,
                scoring="neg_mean_absolute_error",
                random_state=42,
            )
        ],
        "histgradientboostingregressor__learning_rate": [0.01, 0.025, 0.05, 0.1],
        "histgradientboostingregressor__max_iter": [50, 100, 200],
        "histgradientboostingregressor__max_depth": [5, 10, 20],
        "histgradientboostingregressor__l2_regularization": [0, 0.25, 0.5, 0.75, 1],
    },
]
CONFIGURACIONES = {
    "PD": {"objetivo": "target_pd", "solo_default": False},
    "EAD": {"objetivo": "target_ead", "solo_default": True},
    "LGD": {"objetivo": "target_lgd", "solo_default": True},
}
SCORING_REGRESION = make_scorer(mae_acotado, greater_is_better=False)

# ── Separación de la reserva ─────────
# La reserva se identifica por id_cliente: esos registros quedan fuera de la búsqueda y del ajuste inicial.
datos = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8")
datos = prepara_datos(datos)
reserva = pd.read_pickle(VALIDACION_PATH)
if ID in reserva.columns:
    ids_reserva = reserva[ID]
elif ID in reserva.index.names:
    ids_reserva = reserva.index.get_level_values(ID)
else:
    raise ValueError(f"La reserva debe contener {ID} como columna o índice")
ids_reserva = pd.to_numeric(ids_reserva, errors="raise").astype("int64")
ids_datos = pd.to_numeric(datos[ID], errors="raise").astype("int64")
mascara_reserva = ids_datos.isin(set(ids_reserva.tolist()))
datos_entrenamiento = datos.loc[~mascara_reserva].copy()
datos_reserva = datos.loc[mascara_reserva].copy()
objetivos_entrenamiento = crea_objetivos(datos_entrenamiento)
objetivos_reserva = crea_objetivos(datos_reserva)
objetivos_totales = crea_objetivos(datos)

# ── Búsqueda, evaluación y producción ─────────
# Para cada objetivo: búsqueda con CV en entrenamiento, una evaluación en reserva y ajuste final con todos los datos.
for nombre_modelo, configuracion in CONFIGURACIONES.items():
    nombre_objetivo = configuracion["objetivo"]
    # EAD y LGD solo se ajustan con préstamos en default; target_pd no es una variable predictora.
    if configuracion["solo_default"]:
        mascara_entrenamiento = objetivos_entrenamiento["target_pd"] == 1
        mascara_evaluacion = objetivos_reserva["target_pd"] == 1
        mascara_total = objetivos_totales["target_pd"] == 1
    else:
        mascara_entrenamiento = pd.Series(True, index=datos_entrenamiento.index)
        mascara_evaluacion = pd.Series(True, index=datos_reserva.index)
        mascara_total = pd.Series(True, index=datos.index)

    X_entrenamiento = datos_entrenamiento.loc[mascara_entrenamiento, COLUMNAS_PREDICTORAS]
    y_entrenamiento = objetivos_entrenamiento.loc[mascara_entrenamiento, nombre_objetivo]
    X_evaluacion = datos_reserva.loc[mascara_evaluacion, COLUMNAS_PREDICTORAS]
    y_evaluacion = objetivos_reserva.loc[mascara_evaluacion, nombre_objetivo]
    X_total = datos.loc[mascara_total, COLUMNAS_PREDICTORAS]
    y_total = objetivos_totales.loc[mascara_total, nombre_objetivo]

    if nombre_modelo == "PD":
        espacio_parametros = PARAMETROS_PD
        iteraciones = 18
        metrica_busqueda = "roc_auc"
    else:
        espacio_parametros = PARAMETROS_REGRESION
        iteraciones = 20
        metrica_busqueda = SCORING_REGRESION

    busqueda = RandomizedSearchCV(
        estimator=crea_pipeline(nombre_modelo),
        param_distributions=espacio_parametros,
        n_iter=iteraciones,
        cv=3,
        scoring=metrica_busqueda,
        n_jobs=-1,
        pre_dispatch="2*n_jobs",
        random_state=42,
        refit=True,
    )
    busqueda.fit(X_entrenamiento, y_entrenamiento)
    mejor_pipeline = busqueda.best_estimator_

    print(f"{nombre_modelo} mejores hiperparámetros: {busqueda.best_params_}")
    if nombre_modelo == "PD":
        print(f"{nombre_modelo} mejor AUC de CV: {busqueda.best_score_:.6f}")
        prediccion_reserva = mejor_pipeline.predict_proba(X_evaluacion)[:, 1]
        metrica_reserva = roc_auc_score(y_evaluacion, prediccion_reserva)
        print(f"{nombre_modelo} AUC en reserva: {metrica_reserva:.6f}")
    else:
        print(f"{nombre_modelo} mejor MAE de CV: {-busqueda.best_score_:.6f}")
        prediccion_reserva = pd.Series(
            mejor_pipeline.predict(X_evaluacion),
            index=X_evaluacion.index,
        ).clip(lower=0, upper=1)
        metrica_reserva = mean_absolute_error(y_evaluacion, prediccion_reserva)
        print(f"{nombre_modelo} MAE en reserva: {metrica_reserva:.6f}")

    mejor_pipeline.fit(X_total, y_total)
    ruta_artefacto = DIRECTORIO_ARTEFACTOS / f"artefacto_pipeline_{nombre_modelo.lower()}.pkl"
    with open(ruta_artefacto, "wb") as archivo_artefacto:
        cloudpickle.dump(mejor_pipeline, archivo_artefacto)
    print(f"Artefacto guardado: {ruta_artefacto}")
