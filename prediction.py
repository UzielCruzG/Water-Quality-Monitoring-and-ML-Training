import joblib
import numpy as np
import serial
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

# Cargar el modelo desde el archivo
loaded_model = joblib.load('models/modelo_entrenado_Adam_TensorFlow.h5')

# Inicializar la conexión serial con el ESP32
ser = serial.Serial('COM5', 115200)

# Inicializar el MinMaxScaler para normalizar los datos
scaler = MinMaxScaler()

# Esperar un tiempo para que la conexión serial se estabilice
time.sleep(2)

while True:
    # Leer los datos del ESP32
    data = ser.readline().decode().strip()

    # Si hay datos válidos
    if data:
        # Separar los valores y convertirlos en un array de numpy
        values = np.array(data.split(','), dtype=float).reshape(1, -1)

        # Crear un DataFrame temporal para aplicar la normalización
        df_temp = pd.DataFrame(values, columns=['Temperatura', 'TDS', 'Turbidez', 'PH'])

        # Aplicar la normalización a los datos
        values_normalized = scaler.fit_transform(df_temp)

        # Realizar la predicción con el modelo cargado
        y_pred_loaded = loaded_model.predict(values_normalized)

        # Imprimir la predicción
        print("Pertenece a la clase:", y_pred_loaded)


'''
import joblib
import numpy as np

# Cargar el modelo desde el archivo
loaded_model = joblib.load('models/modelo_entrenado_Adam_TensorFlow.h5')

# Ejemplo de dato a predecir
dato = np.array([[0.8215488215488218,0.32244977443120965,0.4427792915531335,0.23529411764705876]])

# Realizar predicciones con el modelo cargado
y_pred_loaded = loaded_model.predict(dato)

print("Pertence a la clase ",y_pred_loaded)
'''