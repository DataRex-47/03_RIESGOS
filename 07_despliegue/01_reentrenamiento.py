# ═══════════════════════════════════════════════════════════════════════
# REENTRENAMIENTO DE LOS TRES MODELOS: PD, EAD y LGD
# ═══════════════════════════════════════════════════════════════════════
#
# Qué hace este archivo, en una frase:
#   para cada modelo busca la mejor receta, mide su nota real una sola vez
#   y deja guardado el modelo final que usa la API.
#
# Mapa del archivo (se lee de arriba abajo, sin saltos):
#   PASO 1  Rutas
#   PASO 2  Listas de columnas
#   PASO 3  Las dos funciones propias que viajan dentro del modelo
#   PASO 4  Cargar los datos, crear los tres objetivos y apartar la reserva
#   PASO 5  El preprocesador (la receta que limpia cada columna)
#   PASO 6  La parrilla: los candidatos que compiten
#   PASO 7  El bucle: PD, EAD y LGD pasan por los mismos 4 pasos
#   PASO 8  Resumen final
#
# Los 4 pasos del bucle (el corazón del archivo):
#   7.1 Elegir filas   7.2 Buscar   7.3 Medir   7.4 Producir

from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


# ═══════════════════════════════════════════════════════════════════════
# PASO 1. RUTAS
# ═══════════════════════════════════════════════════════════════════════
# RAIZ es la carpeta del proyecto (una carpeta más arriba que este archivo).
RAIZ = Path(__file__).resolve().parents[1]

ruta_csv = RAIZ / "02_datos" / "01_Originales" / "prestamos.csv"
ruta_reserva = RAIZ / "02_datos" / "02_Validacion" / "prestamos_val.pkl"
carpeta_salida = RAIZ / "07_despliegue"

# True guarda los tres .pkl nuevos (pisa los anteriores). False solo muestra notas.
GUARDAR = True


# ═══════════════════════════════════════════════════════════════════════
# PASO 2. LISTAS DE COLUMNAS
# ═══════════════════════════════════════════════════════════════════════
# Las 12 columnas que entran al modelo, agrupadas por el tratamiento que reciben.
categoricas = ["ingresos_verificados", "vivienda", "finalidad", "num_cuotas"]  # una columna de 0 y 1 por categoría
ordinales = ["antigüedad_empleo", "rating"]  # categorías con orden (de menos a más)
numericas = ["ingresos", "dti", "num_lineas_credito", "porc_uso_revolving", "principal"]  # se escalan de 0 a 1
binarias = ["num_derogatorios"]  # solo importa si tiene o no tiene
predictoras = categoricas + ordinales + numericas + binarias

# El orden de las categorías ordinales (la posición en la lista es el número que reciben).
orden_antiguedad = [
    "Desconocido", "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
]
orden_rating = ["A", "B", "C", "D", "E", "F", "G"]

# Estados del préstamo que cuentan como default (impago).
estados_default = [
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
    "Default",
]


# ═══════════════════════════════════════════════════════════════════════
# PASO 3. LAS DOS FUNCIONES PROPIAS QUE VIAJAN DENTRO DEL MODELO
# ═══════════════════════════════════════════════════════════════════════
# Estas dos van dentro del pipeline: cuando la API reciba un cliente nuevo,
# las aplicará igual que aquí. Por eso son funciones y no código suelto.

def unifica_categorias(X):
    # Junta categorías muy poco frecuentes con otra parecida.
    X = X.copy()
    X["vivienda"] = X["vivienda"].replace({"ANY": "MORTGAGE", "NONE": "MORTGAGE", "OTHER": "MORTGAGE"})
    X["finalidad"] = X["finalidad"].replace(
        {"wedding": "other", "educational": "other", "renewable_energy": "other"}
    )
    return X


def limita_numericas(X):
    # dti y porcentaje de uso del crédito no pueden salirse de 0 a 100.
    X = X.copy()
    X["dti"] = X["dti"].clip(lower=0, upper=100)
    X["porc_uso_revolving"] = X["porc_uso_revolving"].clip(lower=0, upper=100)
    return X


# ═══════════════════════════════════════════════════════════════════════
# PASO 4. DATOS: CARGAR, CREAR LOS TRES OBJETIVOS Y APARTAR LA RESERVA
# ═══════════════════════════════════════════════════════════════════════
datos = pd.read_csv(ruta_csv)

# Calidad: se descartan los préstamos con ingresos mayores a 400000.
# Los ingresos ausentes cuentan como 0 para esta comparación.
ingresos = pd.to_numeric(datos["ingresos"], errors="coerce").fillna(0)
datos = datos[ingresos <= 400000].copy()

# Los tres objetivos salen de columnas del propio préstamo.
principal = pd.to_numeric(datos["principal"], errors="coerce").fillna(0)
amortizado = pd.to_numeric(datos["imp_amortizado"], errors="coerce").fillna(0)
recuperado = pd.to_numeric(datos["imp_recuperado"], errors="coerce").fillna(0)
pendiente = principal - amortizado  # lo que todavía se debía

# PD: 1 si el préstamo cayó en default, 0 si no.
datos["target_pd"] = datos["estado"].isin(estados_default).astype("int8")

# EAD: qué parte de lo prestado estaba pendiente (entre 0 y 1).
# Si el divisor es 0, la división da vacío y ese vacío se cambia por 0.
datos["target_ead"] = (pendiente / principal.where(principal != 0)).fillna(0).clip(lower=0, upper=1)

# LGD: qué parte de lo pendiente no se logró recuperar (entre 0 y 1).
datos["target_lgd"] = (1 - recuperado / pendiente.where(pendiente != 0)).fillna(0).clip(lower=0, upper=1)

# La reserva (el 30 % apartado desde el notebook 01) se reconoce por id_cliente.
# Nunca se usa para buscar: solo para medir una vez.
reserva_guardada = pd.read_pickle(ruta_reserva)

# El id de la reserva puede venir como columna o como índice.
if "id_cliente" in reserva_guardada.columns:
    ids_reserva = reserva_guardada["id_cliente"]
else:
    ids_reserva = reserva_guardada.index

es_reserva = datos["id_cliente"].isin(ids_reserva)

# Si no se encontró ni una fila de la reserva, algo está mal (archivo equivocado o ids de otro tipo).
# Mejor frenar aquí que descubrirlo después de la búsqueda.
assert es_reserva.sum() > 0, "No se encontró la reserva en los datos: revisa prestamos_val.pkl"

entreno = datos[~es_reserva]  # el 70 %: aquí se busca
reserva = datos[es_reserva]   # el 30 %: aquí se mide
# datos completos: aquí se produce el modelo final


# ═══════════════════════════════════════════════════════════════════════
# PASO 5. EL PREPROCESADOR (la receta que limpia cada grupo de columnas)
# ═══════════════════════════════════════════════════════════════════════
def crea_preprocesador():
    # Categóricas: unir categorías raras, rellenar vacíos con la más común, una columna 0/1 por categoría.
    receta_categoricas = make_pipeline(
        FunctionTransformer(unifica_categorias),
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
    )
    # Ordinales: vacíos pasan a "Desconocido", cada categoría pasa a su número de orden y se escala de 0 a 1.
    receta_ordinales = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Desconocido"),
        OrdinalEncoder(
            categories=[orden_antiguedad, orden_rating],
            handle_unknown="use_encoded_value",
            unknown_value=12,
        ),
        MinMaxScaler(),
    )
    # Numéricas: limitar rangos, vacíos pasan a 0 y se escala de 0 a 1.
    receta_numericas = make_pipeline(
        FunctionTransformer(limita_numericas),
        SimpleImputer(strategy="constant", fill_value=0),
        MinMaxScaler(),
    )
    # Binarias: vacíos pasan a 0; cualquier valor mayor que 0 pasa a 1.
    receta_binarias = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        Binarizer(threshold=0),
    )
    # Cada receta se aplica solo a sus columnas. remainder="drop" descarta cualquier otra.
    return make_column_transformer(
        (receta_categoricas, categoricas),
        (receta_ordinales, ordinales),
        (receta_numericas, numericas),
        (receta_binarias, binarias),
        remainder="drop",
    )


# ═══════════════════════════════════════════════════════════════════════
# PASO 6. LA PARRILLA: LOS CANDIDATOS QUE COMPITEN
# ═══════════════════════════════════════════════════════════════════════
# Una sola definición del modelo de árboles, para que "loss" viva en un único lugar.
#   loss    = lo que cada árbol intenta reducir al aprender (aquí, el error absoluto)
#   scoring = con qué se califica un 10 % apartado para decidir cuándo dejar de agregar árboles
arboles = HistGradientBoostingRegressor(
    loss="absolute_error",
    scoring="neg_mean_absolute_error",
    min_samples_leaf=60,
    random_state=42,
)

# PD: una sola familia (regresión logística). 3 valores de l1_ratio x 6 de C = 18 combinaciones.
parametros_pd = {
    "modelo__l1_ratio": [0, 0.5, 1],
    "modelo__C": [0.01, 0.05, 0.1, 0.25, 0.75, 1],
}

# EAD y LGD: tres familias compiten. Cada bloque es una familia con sus valores.
#   Ridge: 10 combinaciones, Lasso: 5, árboles: 4 x 3 x 3 x 5 = 180. Total 195.
parametros_regresion = [
    {
        "modelo": [Ridge()],
        "modelo__alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    {
        "modelo": [Lasso(max_iter=10000)],
        "modelo__alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
    },
    {
        "modelo": [arboles],
        "modelo__learning_rate": [0.01, 0.025, 0.05, 0.1],
        "modelo__max_iter": [50, 100, 200],
        "modelo__max_depth": [5, 10, 20],
        "modelo__l2_regularization": [0, 0.25, 0.5, 0.75, 1],
    },
]


def mae_acotado(y_real, y_predicho):
    # EAD y LGD son proporciones: una predicción de 1,3 se corta a 1 antes de medir el error.
    return mean_absolute_error(y_real, np.clip(y_predicho, 0, 1))


# ═══════════════════════════════════════════════════════════════════════
# PASO 7. EL BUCLE: PD, EAD y LGD PASAN POR LOS MISMOS 4 PASOS
# ═══════════════════════════════════════════════════════════════════════
resumen = []

for nombre in ["PD", "EAD", "LGD"]:
    print("")
    print("===== ", nombre, " =====")

    # ───────────────────────────────────────────────────────────────────
    # 7.1 ELEGIR FILAS: separar X (las 12 columnas) de y (lo que se predice)
    # ───────────────────────────────────────────────────────────────────
    if nombre == "PD":
        objetivo = "target_pd"
        tabla_entreno = entreno  # PD usa todos los préstamos
        tabla_reserva = reserva
        tabla_total = datos
    else:
        if nombre == "EAD":
            objetivo = "target_ead"
        else:
            objetivo = "target_lgd"
        # EAD y LGD solo tienen sentido en préstamos que cayeron en default
        tabla_entreno = entreno[entreno["target_pd"] == 1]
        tabla_reserva = reserva[reserva["target_pd"] == 1]
        tabla_total = datos[datos["target_pd"] == 1]

    X_entreno = tabla_entreno[predictoras]
    y_entreno = tabla_entreno[objetivo]
    X_reserva = tabla_reserva[predictoras]
    y_reserva = tabla_reserva[objetivo]
    X_total = tabla_total[predictoras]
    y_total = tabla_total[objetivo]
    print("Filas de entrenamiento, reserva y total:", X_entreno.shape[0], X_reserva.shape[0], X_total.shape[0])

    # ───────────────────────────────────────────────────────────────────
    # 7.2 BUSCAR: probar candidatos con validación cruzada, solo con el 70 %
    # ───────────────────────────────────────────────────────────────────
    if nombre == "PD":
        modelo_base = LogisticRegression(solver="saga", max_iter=5000, random_state=42)
        parametros = parametros_pd
        cuantas_combinaciones = 18
        metrica_busqueda = "roc_auc"
        nombre_metrica = "AUC"
    else:
        modelo_base = arboles
        parametros = parametros_regresion
        cuantas_combinaciones = 20
        metrica_busqueda = make_scorer(mae_acotado, greater_is_better=False)
        nombre_metrica = "MAE"

    # El pipeline es un tubo: primero el preprocesador, después el modelo.
    pipeline = Pipeline([("prep", crea_preprocesador()), ("modelo", modelo_base)])

    # cv=3: el 70 % se corta en 3 partes; se entrena con 2 y se califica con la tercera, rotando.
    busqueda = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parametros,
        n_iter=cuantas_combinaciones,
        cv=3,
        scoring=metrica_busqueda,
        n_jobs=-1,
        pre_dispatch="2*n_jobs",
        random_state=42,
        refit=True,  # al terminar deja el mejor pipeline entrenado con todo el 70 %
    )
    busqueda.fit(X_entreno, y_entreno)

    print("Mejor receta encontrada:")
    for clave, valor in busqueda.best_params_.items():
        if clave == "modelo":
            print("   algoritmo =", type(valor).__name__)
        else:
            print("  ", clave.replace("modelo__", ""), "=", valor)

    # sklearn siempre busca el valor más alto, así que guarda el MAE con signo negativo.
    # Para mostrarlo normal se le cambia el signo.
    if nombre == "PD":
        nota_cv = busqueda.best_score_
    else:
        nota_cv = -busqueda.best_score_
    print(nombre_metrica, "en validación cruzada:", round(nota_cv, 4))

    # ───────────────────────────────────────────────────────────────────
    # 7.3 MEDIR: calificar UNA sola vez con la reserva (el 30 % que nunca vio)
    # ───────────────────────────────────────────────────────────────────
    mejor_pipeline = busqueda.best_estimator_
    if nombre == "PD":
        probabilidades = mejor_pipeline.predict_proba(X_reserva)[:, 1]
        nota_reserva = roc_auc_score(y_reserva, probabilidades)
    else:
        predicciones = mejor_pipeline.predict(X_reserva)
        nota_reserva = mae_acotado(y_reserva, predicciones)
    print(nombre_metrica, "en la reserva (la nota oficial):", round(nota_reserva, 4))

    # ───────────────────────────────────────────────────────────────────
    # 7.4 PRODUCIR: reentrenar con todos los datos y guardar el modelo final
    # ───────────────────────────────────────────────────────────────────
    # Este modelo ya vio la reserva, por eso no se vuelve a medir con ella.
    mejor_pipeline.fit(X_total, y_total)
    if GUARDAR:
        ruta_pkl = carpeta_salida / ("artefacto_pipeline_" + nombre.lower() + ".pkl")
        with open(ruta_pkl, "wb") as archivo:
            cloudpickle.dump(mejor_pipeline, archivo)
        print("Guardado:", ruta_pkl)
    else:
        print("GUARDAR = False: no se guardó ningún archivo")

    resumen.append([nombre, nombre_metrica, nota_cv, nota_reserva])


# ═══════════════════════════════════════════════════════════════════════
# PASO 8. RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════
print("")
print("Modelo | Métrica | Validación cruzada | Reserva")
for fila in resumen:
    print(fila[0], "|", fila[1], "|", round(fila[2], 4), "|", round(fila[3], 4))
