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
            loss="absolute_error",  # aprende minimizando el MAE, igual que la nota con la que se mide
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
                loss="absolute_error",
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
# Los 3 modelos (PD, EAD, LGD) pasan por LOS MISMOS 4 PASOS, uno detrás de otro:
#   Paso 1. Elegir filas: separar X (predictoras) e y (objetivo) de tres grupos de datos.
#   Paso 2. Buscar:       probar combinaciones de hiperparámetros con CV, solo con entrenamiento (70 %).
#   Paso 3. Medir:        evaluar UNA sola vez en la reserva (30 %). Esa es la nota del modelo.
#   Paso 4. Producir:     reajustar con todos los datos y guardar el .pkl que usa la API.
#
# Los tres grupos de datos:
#   entrenamiento = todo menos la reserva (para buscar)
#   reserva       = ids de prestamos_val.pkl (para medir, nunca para buscar)
#   total         = entrenamiento + reserva (para el modelo final)

GUARDAR_ARTEFACTOS = True  # False = solo ver métricas, sin tocar los .pkl de producción

# Cómo se llama la métrica de cada modelo (PD es clasificación, EAD y LGD son regresión)
NOMBRE_METRICA = {"PD": "AUC", "EAD": "MAE", "LGD": "MAE"}


# Paso 1: devuelve X e y de un modelo a partir de un grupo de datos.
def elige_filas(nombre_modelo, datos_base, objetivos_base):
    nombre_objetivo = CONFIGURACIONES[nombre_modelo]["objetivo"]
    if CONFIGURACIONES[nombre_modelo]["solo_default"]:
        # EAD y LGD solo tienen sentido en préstamos que cayeron en default
        filas_default = objetivos_base["target_pd"] == 1
        X = datos_base.loc[filas_default, COLUMNAS_PREDICTORAS]
        y = objetivos_base.loc[filas_default, nombre_objetivo]
    else:
        # PD usa todos los préstamos
        X = datos_base[COLUMNAS_PREDICTORAS]
        y = objetivos_base[nombre_objetivo]
    return X, y


# Paso 2: prueba combinaciones con validación cruzada (3 partes) y devuelve la búsqueda terminada.
def busca_hiperparametros(nombre_modelo, X, y):
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
        refit=True,  # al terminar, deja el mejor pipeline reajustado con todo el X, y que recibió
    )
    busqueda.fit(X, y)
    return busqueda


# Paso 2 (resultado): la nota promedio del CV en positivo.
# En EAD y LGD sklearn guarda el MAE con signo negativo (así busca el "máximo"), por eso se invierte.
def nota_cv(nombre_modelo, busqueda):
    if nombre_modelo == "PD":
        return busqueda.best_score_
    return -busqueda.best_score_


# Paso 3: mide el pipeline en la reserva. PD con AUC, EAD y LGD con MAE.
def mide_en_reserva(nombre_modelo, pipeline, X_reserva, y_reserva):
    if nombre_modelo == "PD":
        probabilidades = pipeline.predict_proba(X_reserva)[:, 1]
        return roc_auc_score(y_reserva, probabilidades)
    predicciones = pd.Series(pipeline.predict(X_reserva), index=X_reserva.index)
    predicciones = predicciones.clip(lower=0, upper=1)  # EAD y LGD son proporciones entre 0 y 1
    return mean_absolute_error(y_reserva, predicciones)


# Paso 4: guarda el pipeline ya reajustado con todos los datos.
def guarda_artefacto(nombre_modelo, pipeline):
    ruta_artefacto = DIRECTORIO_ARTEFACTOS / f"artefacto_pipeline_{nombre_modelo.lower()}.pkl"
    with open(ruta_artefacto, "wb") as archivo_artefacto:
        cloudpickle.dump(pipeline, archivo_artefacto)
    print(f"Artefacto guardado: {ruta_artefacto}")


# ── Ejecución: se repite para PD, EAD y LGD ─────────
resumen = []
for nombre_modelo in CONFIGURACIONES:
    metrica = NOMBRE_METRICA[nombre_modelo]
    print(f"\n===== {nombre_modelo} =====")

    # Paso 1. Elegir filas
    X_entrenamiento, y_entrenamiento = elige_filas(nombre_modelo, datos_entrenamiento, objetivos_entrenamiento)
    X_reserva, y_reserva = elige_filas(nombre_modelo, datos_reserva, objetivos_reserva)
    X_total, y_total = elige_filas(nombre_modelo, datos, objetivos_totales)
    print(f"Filas: entrenamiento {len(X_entrenamiento)}, reserva {len(X_reserva)}, total {len(X_total)}")

    # Paso 2. Buscar con CV (solo entrenamiento)
    busqueda = busca_hiperparametros(nombre_modelo, X_entrenamiento, y_entrenamiento)
    mejor_pipeline = busqueda.best_estimator_
    nota_en_cv = nota_cv(nombre_modelo, busqueda)
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")
    print(f"{metrica} en CV: {nota_en_cv:.6f}")

    # Paso 3. Medir una sola vez en la reserva
    nota_en_reserva = mide_en_reserva(nombre_modelo, mejor_pipeline, X_reserva, y_reserva)
    print(f"{metrica} en reserva: {nota_en_reserva:.6f}")

    # Paso 4. Reajustar con todos los datos y guardar
    mejor_pipeline.fit(X_total, y_total)
    if GUARDAR_ARTEFACTOS:
        guarda_artefacto(nombre_modelo, mejor_pipeline)
    else:
        print("GUARDAR_ARTEFACTOS = False: no se guardó ningún .pkl")

    resumen.append((nombre_modelo, metrica, nota_en_cv, nota_en_reserva))

# ── Resumen final ─────────
print("\nModelo | Métrica | CV | Reserva")
for nombre_modelo, metrica, nota_en_cv, nota_en_reserva in resumen:
    print(f"{nombre_modelo} | {metrica} | {nota_en_cv:.4f} | {nota_en_reserva:.4f}")
