# Este código entrena 3 modelos MLP para 3 variables objetivo y calcula la matriz de confusión,
# precisión, sensibilidad y especificidad para cada una de ellas.
# Además, visualiza la curva de pérdida durante el entrenamiento y
# las matrices de confusión para cada variable objetivo.

#20 entrenamiento,40prueba,40Validacion
# Importar las bibliotecas necesarias
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Leer el archivo CSV
df = pd.read_csv('dataset/Categorias Discretas/normalized_data_cat.csv')

# 'X' es el conjunto de características y 'y' son las variables objetivo.
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y_clase = df['Clase_Agua']
y_consumo = df['Consumo_Humano']
y_riego = df['Riego']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train_clase, y_test_clase, y_train_consumo, y_test_consumo, y_train_riego, y_test_riego = train_test_split(X, y_clase, y_consumo, y_riego, test_size=0.2, random_state=42)

# Crear y entrenar la red neuronal para cada variable objetivo
model_clase = MLPClassifier(hidden_layer_sizes=(4, 10, 10, 6), max_iter=100, random_state=42, verbose=True)
model_consumo = MLPClassifier(hidden_layer_sizes=(4, 10, 10, 6), max_iter=100, random_state=42, verbose=True)
model_riego = MLPClassifier(hidden_layer_sizes=(4, 10, 10, 6), max_iter=100, random_state=42, verbose=True)

history_clase = model_clase.fit(X_train, y_train_clase)
history_consumo = model_consumo.fit(X_train, y_train_consumo)
history_riego = model_riego.fit(X_train, y_train_riego)

# Guardar los modelos entrenados en archivos
joblib.dump(model_clase, 'models/MLP 3 ModelsClass/modelo_entrenado_MLP_clase.joblib')
joblib.dump(model_consumo, 'models/MLP 3 ModelsClass/modelo_entrenado_MLP_consumo.joblib')
joblib.dump(model_riego, 'models/MLP 3 ModelsClass/modelo_entrenado_MLP_riego.joblib')

# Realizar predicciones para cada variable objetivo
y_pred_clase = model_clase.predict(X_test)
y_pred_consumo = model_consumo.predict(X_test)
y_pred_riego = model_riego.predict(X_test)

# Calcular la matriz de confusión y precisión para cada variable objetivo
conf_matrix_clase = confusion_matrix(y_test_clase, y_pred_clase)
conf_matrix_consumo = confusion_matrix(y_test_consumo, y_pred_consumo)
conf_matrix_riego = confusion_matrix(y_test_riego, y_pred_riego)

accuracy_clase = accuracy_score(y_test_clase, y_pred_clase)
accuracy_consumo = accuracy_score(y_test_consumo, y_pred_consumo)
accuracy_riego = accuracy_score(y_test_riego, y_pred_riego)


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



# Visualización de la curva de pérdida para cada modelo
plt.figure(figsize=(10, 5))
plt.plot(history_clase.loss_curve_, label='Clase')
plt.plot(history_consumo.loss_curve_, label='Consumo Humano')
plt.plot(history_riego.loss_curve_, label='Riego')
plt.title('Curva de Pérdida')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.legend()
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
