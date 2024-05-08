import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo CSV
df = pd.read_csv('dataset/normalized_data.csv')

# Separar características y etiquetas
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y = df['Clase']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Normalizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir los datos en secuencias temporales para LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 10  # Longitud de la secuencia temporal
X_train_seq = create_sequences(X_train, seq_length)
X_test_seq = create_sequences(X_test, seq_length)

# Construir el modelo LSTM con regularización
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train_seq, y_train[seq_length:], epochs=100, batch_size=32, validation_data=(X_test_seq, y_test[seq_length:]))

# Visualizar el historial de entrenamiento
plt.plot(history.history['loss'], label='Loss de entrenamiento')
plt.plot(history.history['val_loss'], label='Loss de validación')
plt.title('Curva de pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Curva de precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Realizar predicciones en el conjunto de prueba
y_pred_prob = model.predict(X_test_seq)
y_pred = np.argmax(y_pred_prob, axis=1)  # Obtener la clase predicha como el índice de la neurona con mayor probabilidad

# Evaluar el modelo en el conjunto de prueba
accuracy = accuracy_score(y_test[seq_length:], y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test[seq_length:], y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiquetas predichas')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión')
plt.show()

# Guardar el modelo entrenado en un archivo (formato Keras)
model.save('models/modelo_entrenado_LSTM_TensorFlow_100.keras')


#------------------------------------------------------------------------

'''
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
#from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
#model = load_model('models/modelo_entrenado_LSTM_TensorFlow.h5')

# Leer el archivo CSV
df = pd.read_csv('dataset/normalized_data.csv')

# Separar características y etiquetas
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y = df['Clase']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Normalizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir los datos en secuencias temporales para LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 10  # Longitud de la secuencia temporal
X_train_seq = create_sequences(X_train, seq_length)
X_test_seq = create_sequences(X_test, seq_length)

# Construir el modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train_seq, y_train[seq_length:], epochs=100, batch_size=32, validation_data=(X_test_seq, y_test[seq_length:]))

# Visualizar el historial de entrenamiento
plt.plot(history.history['loss'], label='Loss de entrenamiento')
plt.plot(history.history['val_loss'], label='Loss de validación')
plt.title('Curva de pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Curva de precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Realizar predicciones en el conjunto de prueba
y_pred_prob = model.predict(X_test_seq)
y_pred = np.argmax(y_pred_prob, axis=1)  # Obtener la clase predicha como el índice de la neurona con mayor probabilidad

# Evaluar el modelo en el conjunto de prueba
accuracy = accuracy_score(y_test[seq_length:], y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test[seq_length:], y_pred)

# Calcular la sensibilidad y especificidad
true_negatives = conf_matrix[0][0]
false_positives = conf_matrix[0][1]
false_negatives = conf_matrix[1][0]
true_positives = conf_matrix[1][1]

sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)

print(f'Sensibilidad del modelo: {sensitivity * 100:.2f}%')
print(f'Especificidad del modelo: {specificity * 100:.2f}%')

# Guardar el modelo entrenado en un archivo (formato Keras)
model.save('model/modelo_entrenado_LSTM_TensorFlow_1000.keras')
'''



#------------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('models/modelo_entrenado_LSTM_TensorFlow.h5')

# Leer el archivo CSV
df = pd.read_csv('dataset/normalized_data.csv')

# Separar características y etiquetas
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y = df['Clase']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Normalizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir los datos en secuencias temporales para LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 10  # Longitud de la secuencia temporal
X_train_seq = create_sequences(X_train, seq_length)
X_test_seq = create_sequences(X_test, seq_length)

# Construir el modelo LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train_seq, y_train[seq_length:], epochs=10, batch_size=32, validation_data=(X_test_seq, y_test[seq_length:]))

# Realizar predicciones en el conjunto de prueba
y_pred_prob = model.predict(X_test_seq)
y_pred = np.argmax(y_pred_prob, axis=1)  # Obtener la clase predicha como el índice de la neurona con mayor probabilidad

# Evaluar el modelo en el conjunto de prueba
accuracy = accuracy_score(y_test[seq_length:], y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test[seq_length:], y_pred)

# Calcular la sensibilidad y especificidad
true_negatives = conf_matrix[0][0]
false_positives = conf_matrix[0][1]
false_negatives = conf_matrix[1][0]
true_positives = conf_matrix[1][1]

sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)

print(f'Sensibilidad del modelo: {sensitivity * 100:.2f}%')
print(f'Especificidad del modelo: {specificity * 100:.2f}%')

# Entrenar el modelo
history = model.fit(X_train_seq, y_train[seq_length:], epochs=10, batch_size=32, validation_data=(X_test_seq, y_test[seq_length:]))

# Guardar el modelo entrenado en un archivo (formato Keras)
model.save('model/modelo_entrenado_LSTM_TensorFlow.h5')
'''