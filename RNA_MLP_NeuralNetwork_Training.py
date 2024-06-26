# Este código entrena 1 modelo MLP con 4 caracteristicas principales de calidad del agua
# y 1 variable objetivo "Clase", a partir de ello, calcula la matriz de confusión,
# precisión, sensibilidad y especificidad para cada una de ellas.

# Importar las bibliotecas necesarias
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Leer el archivo CSV
df = pd.read_csv('dataset/normalized_data.csv')

# 'X' es el conjunto de características y 'y' es la variable objetivo.
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y = df['Clase']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Crear y entrenar la red neuronal
model = MLPClassifier(hidden_layer_sizes=(4, 10, 10, 5), max_iter=100 , random_state=42, verbose=True)
model.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo
joblib.dump(model, 'models/modelo_entrenado_MLP.joblib')

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Calcular la sensibilidad y especificidad
true_negatives = conf_matrix[0][0]
false_negatives = conf_matrix[1][0]
true_positives = conf_matrix[1][1]
false_positives = conf_matrix[0][1]

sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)

print(f'Precisión del modelo: {accuracy * 100:.2f}%')
print(f'Sensibilidad del modelo: {sensitivity * 100:.2f}%')
print(f'Especificidad del modelo: {specificity * 100:.2f}%')
