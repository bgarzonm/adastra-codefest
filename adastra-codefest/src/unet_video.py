# Importando las librerías necesarias
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import matplotlib.pyplot as plt

# Definición de la función para cargar y preprocesar los datos
def load_and_preprocess_data(train_dir):
    img_size=(224, 224)  # Tamaño de las imágenes
    train_data = []  # Lista para almacenar los datos de entrenamiento
    train_labels = []  # Lista para almacenar las etiquetas de entrenamiento

    # Bucle sobre todas las imágenes en el directorio
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg"):  # Si el archivo es una imagen
            img_path = os.path.join(train_dir, filename)  # Ruta de la imagen
            
            # Carga la imagen, cambia su tamaño a uno fijo y la almacena en la lista de datos
            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)
            train_data.append(image)

            # Busca el archivo .txt correspondiente
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(train_dir, txt_filename)

            # Verifica si existe el archivo .txt
            if not os.path.exists(txt_path):
                print(f"Advertencia: No se encontró el archivo de texto para la imagen {filename}")
                continue

            with open(txt_path, 'r') as f:  # Abre el archivo .txt
                lines = f.readlines()  # Lee todas las líneas del archivo

            labels = []  # Lista para almacenar las etiquetas
            for line in lines:
                # Analiza los datos de la etiqueta de cada línea en el archivo de texto (formato YOLO)
                label_data = line.strip().split(' ')
                class_id = int(label_data[0])
                x_center = float(label_data[1])
                y_center = float(label_data[2])
                width = float(label_data[3])
                height = float(label_data[4])

                # Convierte las coordenadas de YOLO a coordenadas normalizadas
                x = x_center * img_size[0]
                y = y_center * img_size[1]
                w = width * img_size[0]
                h = height * img_size[1]

                # Crear un diccionario para almacenar los detalles de las etiquetas
                label = {
                    'class_id': class_id,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
                labels.append(label)  # Añadir etiquetas a la lista
            
            train_labels.append(labels)  # Añadir etiquetas a la lista de etiquetas de entrenamiento

            # Imprimir la información de la imagen y de la etiqueta procesadas
            print("Imagen:", filename)
            print("Etiquetas:", labels)
            print("-----------------------------------")

    # Convertir los datos de entrenamiento y las etiquetas a arrays numpy
    train_data = np.array(train_data, dtype="float32")
    train_labels = np.array(train_labels)

    # Normalizar los datos de entrenamiento
    train_data = train_data / 255.0

    # Imprimir el número total de imágenes cargadas
    print("Número total de imágenes:", len(train_data))

    return train_data, train_labels

load_and_preprocess_data('/kaggle/input/last-attempt')  # Llamando a la función con la ruta del directorio

# Definición de la arquitectura del modelo U-Net
def unet(input_shape):
    inputs = Input(input_shape)
    
    # Camino de contracción
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Camino de expansión
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Capa de salida
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)  # Definición del modelo
    return model

# Crear el modelo U-Net
input_shape = (img_size[0], img_size[1], 3)
model = unet(input_shape)  # Inicializar el modelo U-Net con el tamaño de entrada especificado

# Función para procesar las etiquetas de entrenamiento
def preprocess_labels(train_labels, img_size):
    train_labels_processed = []
    for labels in train_labels:
        mask = np.zeros((img_size[0], img_size[1], 1), dtype=np.float32)  # Crear una máscara de ceros
        for label in labels:
            x = int(label['x'] * img_size[1])
            y = int(label['y'] * img_size[0])
            width = int(label['width'] * img_size[1])
            height = int(label['height'] * img_size[0])
            class_id = 1  # Establecer el ID de clase como 1 para simplificar

            mask[y:y+height, x:x+width] = class_id  # Rellenar el área delimitada por la etiqueta con el ID de clase

        train_labels_processed.append(mask)  # Agregar la máscara procesada a la lista

    train_labels_processed = np.array(train_labels_processed)  # Convertir la lista en un array numpy

    return train_labels_processed

# Función para entrenar el modelo
def train_model(model, train_data, train_labels, img_size, batch_size=32, epochs=10, validation_split=0.2):
    train_labels_processed = preprocess_labels(train_labels, img_size)  # Procesar las etiquetas de entrenamiento
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_data, test_data, train_labels_processed, test_labels_processed = train_test_split(train_data, train_labels_processed, test_size=0.2, random_state=42)

    # Entrenar el modelo
    history = model.fit(train_data, train_labels_processed, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    return history

history = train_model(model, train_data, train_labels, img_size, batch_size=32, epochs=10, validation_split=0.2)  # Llamando a la función para entrenar el modelo

# Función para visualizar la historia de entrenamiento
def plot_training_history(history):
    # Graficar la pérdida de entrenamiento y validación
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Pérdida en Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida en Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida en Entrenamiento y Validación')
    plt.legend()
    plt.show()

    # Graficar la precisión de entrenamiento y validación
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Precisión en Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión en Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Precisión en Entrenamiento y Validación')
    plt.legend()
    plt.show()

plot_training_history(history)  # Llamando a la función para graficar la historia de entrenamiento
