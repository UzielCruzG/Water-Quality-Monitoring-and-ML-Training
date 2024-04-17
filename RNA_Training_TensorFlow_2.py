import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

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

# Construir el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  # Capa de salida con 6 neuronas para las seis clases
])

# Compilar el modelo con el optimizador "adam" y la función de pérdida "sparse_categorical_crossentropy"
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con un número mayor de épocas
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Guardar el modelo entrenado en un archivo (formato Keras nativo)
model.save('models/modelo_entrenado_TensorFlow.h5')

# Realizar predicciones en el conjunto de prueba
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Obtener la clase predicha como el índice de la neurona con mayor probabilidad

# Evaluar el modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Calcular y visualizar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiquetas predichas')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión')
plt.show()

# Calcular la sensibilidad y especificidad
num_classes = len(np.unique(y_test))
sens_spec = {}
for i in range(num_classes):
    true_negatives = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))
    false_positives = np.sum(np.delete(conf_matrix[i, :], i))
    false_negatives = np.sum(np.delete(conf_matrix[:, i], i))
    true_positives = conf_matrix[i, i]
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    sens_spec[f'Clase {i}'] = {'Sensibilidad': sensitivity, 'Especificidad': specificity}

# Imprimir sensibilidad y especificidad para cada clase
for clase, metrics in sens_spec.items():
    print(f'{clase}: Sensibilidad={metrics["Sensibilidad"]}, Especificidad={metrics["Especificidad"]}')
