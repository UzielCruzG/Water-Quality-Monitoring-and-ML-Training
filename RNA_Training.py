# Importar las bibliotecas necesarias
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Leer el archivo CSV
df = pd.read_csv('dataset/normalized_data.csv')

# 'X' es el conjunto de características y 'y' es la variable objetivo.
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y = df['Clase']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Crear y entrenar la red neuronal
model = MLPClassifier(hidden_layer_sizes=(4, 10, 10, 5), max_iter=1000, random_state=42, verbose=True)
model.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo
joblib.dump(model, 'modelo_entrenado.joblib')

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
