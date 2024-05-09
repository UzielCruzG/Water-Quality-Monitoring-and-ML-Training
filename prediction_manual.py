import joblib
import numpy as np

# Cargar el modelo desde el archivo
loaded_model = joblib.load('models/modelo_entrenado_MLP.joblib')

# Ejemplo de dato a predecir
dato = np.array([[0.84175084, 0.47401902, 0.00376022, 0.42180775]])

# Realizar predicciones con el modelo cargado
y_pred_loaded = loaded_model.predict(dato)

print("Pertence a la clase ",y_pred_loaded)
