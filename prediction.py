import joblib
import numpy as np
import serial
import pandas as pd
import time
#import tensorflow as tf

# Cargar el modelo desde el archivo
loaded_model = joblib.load('models/modelo_entrenado_MLP.joblib')
#loaded_model = tf.keras.models.load_model('models/modelo_entrenado_Adam_TensorFlow.h5')

# Inicializar la conexión serial con el ESP32
ser = serial.Serial('COM5', 115200)

# Definir los valores máximos y mínimos para normalizar los datos
max_values = np.array([28.5, 327.12, 367, 11.68])
min_values = np.array([22.56, 19.01, 0, 4.71])

# Esperar un tiempo para que la conexión serial se estabilice
time.sleep(1)

while True:
    # Leer los datos del ESP32
    data = ser.readline().decode().strip()
    print(data)

    # Si hay datos válidos
    if data and ',' in data:
        # Separar los valores y convertirlos en un array de numpy
        values = np.array(data.split(','), dtype=float)

        # Normalizar manualmente los datos
        values_normalized = (values - min_values) / (max_values - min_values)
        print(values_normalized)

        # Realizar la predicción con el modelo cargado
        y_pred_loaded = loaded_model.predict(values_normalized.reshape(1, -1))

        # Imprimir la predicción
        print("Pertenece a la clase:", y_pred_loaded)

'''
import joblib
import numpy as np
import serial
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
#import tensorflow as tf

# Cargar el modelo desde el archivo
loaded_model = joblib.load('models/modelo_entrenado_MLP.joblib')
#loaded_model = tf.keras.models.load_model('models/modelo_entrenado_Adam_TensorFlow.h5')

# Inicializar la conexión serial con el ESP32
ser = serial.Serial('COM5', 115200)

# Definir los valores máximos y mínimos para normalizar los datos
max_values = np.array([28.5, 327.12, 367, 11.68])
min_values = np.array([22.56, 19.01, 0, 4.71])

# Concatenar los valores máximos y mínimos en un solo array
feature_range = np.vstack((min_values, max_values))

# Inicializar el MinMaxScaler con los valores máximos y mínimos y ajustarlo
scaler = MinMaxScaler(feature_range=(min_values.min(), max_values.max()))
scaler.fit(feature_range)

# Esperar un tiempo para que la conexión serial se estabilice
time.sleep(3)

while True:
    # Leer los datos del ESP32
    data = ser.readline().decode().strip()
    print(data)

    # Si hay datos válidos
    if data and ',' in data:
        # Separar los valores y convertirlos en un array de numpy
        values = np.array(data.split(','), dtype=float).reshape(1, -1)

        # Normalizar los datos usando el MinMaxScaler ajustado
        values_normalized = scaler.transform(values)
        print(values_normalized)

        # Realizar la predicción con el modelo cargado
        y_pred_loaded = loaded_model.predict(values_normalized)

        # Imprimir la predicción
        print("Pertenece a la clase:", y_pred_loaded)

'''


# # Comunicación por bluetooth
# import bluetooth

# # Dirección MAC del dispositivo bluetooth
# mac_address = "XX:XX:XX:XX:XX:XX"  # Reemplaza con la dirección MAC correcta

# # Inicializar el socket
# sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

# # Conectar con el dispositivo
# sock.connect((mac_address, 1))

# while True:
#     # Leer los datos del dispositivo bluetooth
#     data = sock.recv(1024).decode().strip()

#     # Si hay datos válidos
#     if data:
#         # Separar los valores y convertirlos en un array de numpy
#         values = np.array(data.split(','), dtype=float).reshape(1, -1)

#         # Crear un DataFrame temporal para aplicar la normalización
#         df_temp = pd.DataFrame(values, columns=['Temperatura', 'TDS', 'Turbidez', 'PH'])

#         # Aplicar la normalización a los datos
#         values_normalized = scaler.fit_transform(df_temp)

#         # Realizar la predicción con el modelo cargado
#         y_pred_loaded = loaded_model.predict(values_normalized)

#         # Imprimir la predicción
#         print("Pertenece a la clase:", y_pred_loaded)

# # Cerrar el socket
# sock.close()
