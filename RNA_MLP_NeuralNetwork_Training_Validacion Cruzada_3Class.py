# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Leer el archivo CSV
df = pd.read_csv('dataset/Categorias Discretas/normalized_data_cat.csv')

# 'X' es el conjunto de características y 'y' son las variables objetivo.
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y_clase = df['Clase_Agua']
y_consumo = df['Consumo_Humano']
y_riego = df['Riego']

# Función para trazar la curva de pérdida
def plot_loss_curve(model, X, y, cv):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Curva de Pérdida')
    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Pérdida')
    for train_index, val_index in cv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        ax.plot(model.loss_curve_, label=f'Fold {len(ax.lines) + 1}')
    ax.legend()
    plt.show()

# Función para calcular los resultados de cada fold de la validación cruzada
def calculate_cv_results(model, X, y, cv):
    cv_results = []
    y_true_all = []
    y_pred_all = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        cv_results.append(accuracy)
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    return cv_results, conf_matrix

# Función para graficar la matriz de confusión
def plot_confusion_matrix(conf_matrix, variable_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión para {variable_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Crear los modelos de red neuronal
model_clase = MLPClassifier(hidden_layer_sizes=(4, 5, 5, 6), max_iter=1000, random_state=42)
model_consumo = MLPClassifier(hidden_layer_sizes=(4, 8, 7, 6), max_iter=1000, random_state=42)
model_riego = MLPClassifier(hidden_layer_sizes=(4, 5, 5, 6), max_iter=1000, random_state=42)

# Realizar la validación cruzada con KFold
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Visualizar la curva de pérdida para cada modelo
print("Curva de Pérdida para la variable objetivo 'Clase_Agua':")
plot_loss_curve(model_clase, X, y_clase, cv)

print("Curva de Pérdida para la variable objetivo 'Consumo_Humano':")
plot_loss_curve(model_consumo, X, y_consumo, cv)

print("Curva de Pérdida para la variable objetivo 'Riego':")
plot_loss_curve(model_riego, X, y_riego, cv)

# Calcular los resultados de cada fold de la validación cruzada y obtener la matriz de confusión
cv_results_clase, conf_matrix_clase = calculate_cv_results(model_clase, X, y_clase, cv)
cv_results_consumo, conf_matrix_consumo = calculate_cv_results(model_consumo, X, y_consumo, cv)
cv_results_riego, conf_matrix_riego = calculate_cv_results(model_riego, X, y_riego, cv)

# Mostrar los resultados de cada fold de la validación cruzada
def print_cv_results(cv_results, variable_name):
    print(f"Resultados de la validación cruzada para la variable objetivo '{variable_name}':")
    for i, accuracy in enumerate(cv_results, start=1):
        print(f"Fold {i}: Accuracy: {accuracy:.2f}")
    print(f'Precisión media para {variable_name}: {np.mean(cv_results) * 100:.2f}%')

print_cv_results(cv_results_clase, 'Clase_Agua')
print_cv_results(cv_results_consumo, 'Consumo_Humano')
print_cv_results(cv_results_riego, 'Riego')

# Graficar las matrices de confusión
plot_confusion_matrix(conf_matrix_clase, 'Clase_Agua')
plot_confusion_matrix(conf_matrix_consumo, 'Consumo_Humano')
plot_confusion_matrix(conf_matrix_riego, 'Riego')



'''
# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Leer el archivo CSV
df = pd.read_csv('dataset/Categorias Discretas/normalized_data_cat.csv')

# 'X' es el conjunto de características y 'y' son las variables objetivo.
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y_clase = df['Clase_Agua']
y_consumo = df['Consumo_Humano']
y_riego = df['Riego']


# Función para trazar la curva de pérdida
def plot_loss_curve(model, X, y, cv):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Curva de Pérdida')
    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Pérdida')
    for train_index, val_index in cv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        ax.plot(model.loss_curve_, label=f'Fold {len(ax.lines) + 1}')
    ax.legend()
    plt.show()

# Función para calcular los resultados de cada fold de la validación cruzada
def calculate_cv_results(model, X, y, cv):
    cv_results = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        cv_results.append({'Accuracy': accuracy, 'Confusion Matrix': conf_matrix})
    return cv_results


# Crear y entrenar la red neuronal para cada variable objetivo usando validación cruzada
model_clase = MLPClassifier(hidden_layer_sizes=(4, 5, 5, 6), max_iter=100, random_state=42)
model_consumo = MLPClassifier(hidden_layer_sizes=(4, 6, 6, 6), max_iter=100, random_state=42)
model_riego = MLPClassifier(hidden_layer_sizes=(4, 5, 5, 6), max_iter=100, random_state=42)

# Realizar validación cruzada para obtener la precisión media
cv_scores_clase = cross_val_score(model_clase, X, y_clase, cv=5)
cv_scores_consumo = cross_val_score(model_consumo, X, y_consumo, cv=5)
cv_scores_riego = cross_val_score(model_riego, X, y_riego, cv=5)

# Imprimir los resultados de la validación cruzada
print("Resultados de la validación cruzada para la variable objetivo 'Clase':")
print(f'Precisión media: {np.mean(cv_scores_clase) * 100:.2f}%')

print("Resultados de la validación cruzada para la variable objetivo 'Consumo_Humano':")
print(f'Precisión media: {np.mean(cv_scores_consumo) * 100:.2f}%')

print("Resultados de la validación cruzada para la variable objetivo 'Riego':")
print(f'Precisión media: {np.mean(cv_scores_riego) * 100:.2f}%')


# Visualización de la curva de pérdida para cada modelo
plt.figure(figsize=(10, 5))
plt.plot(model_clase.loss_curve_, label='Clase')
plt.plot(model_consumo.loss_curve_, label='Consumo Humano')
plt.plot(model_riego.loss_curve_, label='Riego')
plt.title('Curva de Pérdida')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Visualización de las matrices de confusión
plt.figure(figsize=(15, 10))

# Matriz de Confusión para la variable objetivo 'Clase'
plt.subplot(2, 3, 1)
sns.heatmap(confusion_matrix(y_clase, cross_val_predict(model_clase, X, y_clase, cv=5)), annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Matriz de Confusión (Clase)')
plt.xlabel('Predicción')
plt.ylabel('Real')

# Matriz de Confusión para la variable objetivo 'Consumo_Humano'
plt.subplot(2, 3, 2)
sns.heatmap(confusion_matrix(y_consumo, cross_val_predict(model_consumo, X, y_consumo, cv=5)), annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Matriz de Confusión (Consumo Humano)')
plt.xlabel('Predicción')
plt.ylabel('Real')

# Matriz de Confusión para la variable objetivo 'Riego'
plt.subplot(2, 3, 3)
sns.heatmap(confusion_matrix(y_riego, cross_val_predict(model_riego, X, y_riego, cv=5)), annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Matriz de Confusión (Riego)')
plt.xlabel('Predicción')
plt.ylabel('Real')

plt.tight_layout()
plt.show()
'''

#-------------------------------------------------------------------------------------------------------------



'''
# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Leer el archivo CSV
df = pd.read_csv('dataset/Categorias Discretas/normalized_data_cat.csv')

# 'X' es el conjunto de características y 'y' son las variables objetivo.
X = df[['Temperatura', 'TDS', 'Turbidez', 'PH']]
y_clase = df['Clase_Agua']
y_consumo = df['Consumo_Humano']
y_riego = df['Riego']

# Función para trazar la curva de pérdida
def plot_loss_curve(model, X, y, cv):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Curva de Pérdida')
    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Pérdida')
    for train_index, val_index in cv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        ax.plot(model.loss_curve_, label=f'Fold {len(ax.lines) + 1}')
    ax.legend()
    plt.show()

# Función para calcular los resultados de cada fold de la validación cruzada
def calculate_cv_results(model, X, y, cv):
    cv_results = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        cv_results.append({'Accuracy': accuracy, 'Confusion Matrix': conf_matrix})
    return cv_results

# Crear los modelos de red neuronal
model_clase = MLPClassifier(hidden_layer_sizes=(4, 5, 5, 6), max_iter=1000, random_state=42)
model_consumo = MLPClassifier(hidden_layer_sizes=(4, 5, 5, 5), max_iter=1000, random_state=42)
model_riego = MLPClassifier(hidden_layer_sizes=(4, 5, 5, 6), max_iter=1000, random_state=42)

# Realizar la validación cruzada con KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Visualizar la curva de pérdida para cada modelo
print("Curva de Pérdida para la variable objetivo 'Clase':")
plot_loss_curve(model_clase, X, y_clase, cv)

print("Curva de Pérdida para la variable objetivo 'Consumo_Humano':")
plot_loss_curve(model_consumo, X, y_consumo, cv)

print("Curva de Pérdida para la variable objetivo 'Riego':")
plot_loss_curve(model_riego, X, y_riego, cv)

# Calcular los resultados de cada fold de la validación cruzada
cv_results_clase = calculate_cv_results(model_clase, X, y_clase, cv)
cv_results_consumo = calculate_cv_results(model_consumo, X, y_consumo, cv)
cv_results_riego = calculate_cv_results(model_riego, X, y_riego, cv)

# Mostrar los resultados de cada fold de la validación cruzada
print("Resultados de la validación cruzada para la variable objetivo 'Clase':")
for i, result in enumerate(cv_results_clase, start=1):
    print(f"Fold {i}:")
    print(f"Accuracy: {result['Accuracy']}")
    print("Confusion Matrix:")
    print(result['Confusion Matrix'])
    print()

print("Resultados de la validación cruzada para la variable objetivo 'Consumo_Humano':")
for i, result in enumerate(cv_results_consumo, start=1):
    print(f"Fold {i}:")
    print(f"Accuracy: {result['Accuracy']}")
    print("Confusion Matrix:")
    print(result['Confusion Matrix'])
    print()

print("Resultados de la validación cruzada para la variable objetivo 'Riego':")
for i, result in enumerate(cv_results_riego, start=1):
    print(f"Fold {i}:")
    print(f"Accuracy: {result['Accuracy']}")
    print("Confusion Matrix:")
    print(result['Confusion Matrix'])
    print()

'''