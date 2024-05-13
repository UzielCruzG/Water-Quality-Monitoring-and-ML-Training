#Este código entrena un modelo MLP con 3 variables objetivo y calcula la matriz de confusión, precisión, sensibilidad y especificidad para cada una de ellas. Además, visualiza la curva de pérdida durante el entrenamiento y las matrices de confusión para cada variable objetivo.
# Importar las bibliotecas necesarias
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Leer el archivo CSV
df = pd.read_csv('dataset/new_normalized_data.csv')

# 'X' es el conjunto de características y 'y' son las variables objetivo.
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y = df[['Clase_Agua', 'Consumo_Humano', 'Riego']]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Crear y entrenar la red neuronal
model = MLPClassifier(hidden_layer_sizes=(4, 10, 10, 5), max_iter=100, random_state=42, verbose=True)
history = model.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo
joblib.dump(model, 'models/new_modelo_entrenado_MLP.joblib')

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular la matriz de confusión y precisión para cada variable objetivo
conf_matrix_clase = confusion_matrix(y_test['Clase'], y_pred[:, 0])
conf_matrix_consumo = confusion_matrix(y_test['Consumo_Humano'], y_pred[:, 1])
conf_matrix_riego = confusion_matrix(y_test['Riego'], y_pred[:, 2])

accuracy_clase = accuracy_score(y_test['Clase'], y_pred[:, 0])
accuracy_consumo = accuracy_score(y_test['Consumo_Humano'], y_pred[:, 1])
accuracy_riego = accuracy_score(y_test['Riego'], y_pred[:, 2])

# Calcular la sensibilidad y especificidad para cada variable objetivo
sensitivity_clase = conf_matrix_clase[1, 1] / (conf_matrix_clase[1, 0] + conf_matrix_clase[1, 1])
specificity_clase = conf_matrix_clase[0, 0] / (conf_matrix_clase[0, 0] + conf_matrix_clase[0, 1])

sensitivity_consumo = conf_matrix_consumo[1, 1] / (conf_matrix_consumo[1, 0] + conf_matrix_consumo[1, 1])
specificity_consumo = conf_matrix_consumo[0, 0] / (conf_matrix_consumo[0, 0] + conf_matrix_consumo[0, 1])

sensitivity_riego = conf_matrix_riego[1, 1] / (conf_matrix_riego[1, 0] + conf_matrix_riego[1, 1])
specificity_riego = conf_matrix_riego[0, 0] / (conf_matrix_riego[0, 0] + conf_matrix_riego[0, 1])


# Imprimir resultados para cada variable objetivo
print("Resultados para la variable objetivo 'Clase':")
print(f'Precisión del modelo: {accuracy_clase * 100:.2f}%')
print(f'Sensibilidad del modelo: {sensitivity_clase * 100:.2f}%')
print(f'Especificidad del modelo: {specificity_clase * 100:.2f}%')

print("Resultados para la variable objetivo 'Consumo_Humano':")
print(f'Precisión del modelo: {accuracy_consumo * 100:.2f}%')
print(f'Sensibilidad del modelo: {sensitivity_consumo * 100:.2f}%')
print(f'Especificidad del modelo: {specificity_consumo * 100:.2f}%')

print("Resultados para la variable objetivo 'Riego':")
print(f'Precisión del modelo: {accuracy_riego * 100:.2f}%')
print(f'Sensibilidad del modelo: {sensitivity_riego * 100:.2f}%')
print(f'Especificidad del modelo: {specificity_riego * 100:.2f}%')



# Visualización de la curva de pérdida
plt.figure(figsize=(10, 5))
plt.plot(model.loss_curve_)
plt.title('Curva de Pérdida')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.show()

# Visualización de las matrices de confusión
plt.figure(figsize=(15, 10))

# Matriz de Confusión para la variable objetivo 'Clase'
plt.subplot(2, 3, 1)
sns.heatmap(conf_matrix_clase, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Matriz de Confusión (Clase)')
plt.xlabel('Predicción')
plt.ylabel('Real')

# Matriz de Confusión para la variable objetivo 'Consumo_Humano'
plt.subplot(2, 3, 2)
sns.heatmap(conf_matrix_consumo, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Matriz de Confusión (Consumo Humano)')
plt.xlabel('Predicción')
plt.ylabel('Real')

# Matriz de Confusión para la variable objetivo 'Riego'
plt.subplot(2, 3, 3)
sns.heatmap(conf_matrix_riego, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Matriz de Confusión (Riego)')
plt.xlabel('Predicción')
plt.ylabel('Real')

plt.tight_layout()
plt.show()


#--------------------------------------------------------------------------------------------------------------------------
#El siguiente codigo fue usado para entrenamiento de un modelo MLP usando solo 1 variable objetivo (clasificacion de tipo de agua)
'''
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
'''