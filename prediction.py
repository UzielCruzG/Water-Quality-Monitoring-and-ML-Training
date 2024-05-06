import joblib
import numpy as np

# Cargar el modelo desde el archivo
loaded_model = joblib.load('models/modelo_entrenado.joblib')

# Ejemplo de dato a predecir
dato = np.array([[0.8215488215488218,0.32244977443120965,0.4427792915531335,0.23529411764705876]])

# Realizar predicciones con el modelo cargado
y_pred_loaded = loaded_model.predict(dato)

print("Pertence a la clase ",y_pred_loaded)
