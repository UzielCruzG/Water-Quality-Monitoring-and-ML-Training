Water Quality Monitoring and ML Training

Este repositorio contiene programas de entrenamiento en Python para modelos de aprendizaje automático destinados al análisis de la calidad del agua. Los modelos se entrenan utilizando datos recopilados de sensores de temperatura, TDS, turbidez y pH, que están disponibles en el conjunto de datos proporcionado en la carpeta dataset.
Estructura del Repositorio

    dataset: Esta carpeta contiene el archivo de datos de los sensores referentes a la calidad del agua. Incluye tanto los datos crudos sin procesar en el archivo Datos_Sensores.csv, como los datos transformados en el archivo normalized_data.csv.
    models: En esta carpeta se encuentran los programas de entrenamiento para los modelos de aprendizaje automático.    
    RNA_MLP_NeuralNetwork_Training.py: Programa de entrenamiento para un modelo de Perceptrón Multicapa (MLP) utilizando la biblioteca Scikit-learn.
    TensorFlow_Adam_NeuralNetwork_Training.py: Programa de entrenamiento para un modelo de red neuronal utilizando TensorFlow con el optimizador Adam.
    normalize.py: Script Python para normalizar los datos del conjunto de datos.
    prediction.py: Script Python para realizar predicciones de clase utilizando los modelos entrenados.

Uso

    Clona el repositorio a tu entorno local.
    Utiliza los archivos en la carpeta dataset para cargar los datos de los sensores.
    Ejecuta los programas de entrenamiento en la carpeta models para entrenar los modelos de aprendizaje automático.
    Utiliza el script prediction.py para realizar predicciones de clasificación utilizando los modelos entrenados.

Requisitos del Entorno

    Python
    Bibliotecas Python: Scikit-learn, TensorFlow, Pandas, NumPy, Matplotlib, Seaborn.

    Para hacer uso de la libreria TensorFlow tomar en cuenta: 
        1. Controladores de GPU NVIDIA®: CUDA® 11.2 requiere la versión 450.80.02 o una posterior.
        2. En caso de usar CPU, TensorFlow usa operaciones vectoriales como SSE o AVX, así que debes tener una CPU compatible
