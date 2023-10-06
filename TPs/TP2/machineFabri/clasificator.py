from joblib import load

# carga el modelo
clasificador = load('filename.joblib')

etiquetaPredicha = clasificador.predict(invariantesDeHu)
