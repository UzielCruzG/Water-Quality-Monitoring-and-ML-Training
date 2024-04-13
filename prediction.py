import joblib
import numpy as np

# Cargar el modelo desde el archivo
loaded_model = joblib.load('modelo_entrenado.joblib')

# Ejemplo de dato a predecir
dato = np.array([[0.10000,0.01636,0.92561,0.95500]]) #Cambiar a que python lea los valores en tiempo real

# Realizar predicciones con el modelo cargado
y_pred_loaded = loaded_model.predict(dato)

print(y_pred_loaded)

#Recuerda cambiar la entrada d evalores, normalizar los datos en tiempo real.