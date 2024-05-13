import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Leer el archivo CSV
df = pd.read_csv("dataset/Registros_Sensores_New.csv")

# Lista de columnas a normalizar
columns_to_normalize = ['Temperatura', 'TDS', 'Turbidez', 'PH']

# Inicializar el MinMaxScaler
scaler = MinMaxScaler()

# Aplicar la normalización a las columnas seleccionadas
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Guardar los datos normalizados en un nuevo archivo CSV
df.to_csv('dataset/new_normalized_data.csv', index=False)
