# CODEFEST 2023

Aquí encontrarás las instrucciones sobre cómo descargar el archivo `requirements.txt` y cómo utilizar dos archivos en el repositorio.

## Descargar `requirements.txt`

Para descargar el archivo `requirements.txt`, sigue los siguientes pasos:

1. Asegúrate de tener instalado Python en tu sistema.
2. Abre una terminal o línea de comandos.
3. Navega hasta el directorio donde se encuentra el repositorio.
4. Ejecuta el siguiente comando:

pip install -r requirements.txt

Esto instalará todas las dependencias especificadas en el archivo `requirements.txt`.

## Utilizar dos archivos en el repositorio

Para utilizar dos archivos en el repositorio, sigue los siguientes pasos:

1. Asegúrate de haber descargado y configurado todas las dependencias del proyecto siguiendo las instrucciones anteriores.
2. Identifica los dos archivos que deseas utilizar en el repositorio.
3. Puedes importar estos archivos en tu código Python utilizando la siguiente sintaxis:
4. 

```python
from news_ner import *
from unet_video import *

ner_from_file(text_path, output_path)
ner_from_url(url,path file json)
ner_from_str(text, output_path)

train_dir = "PATH"
img_size = (int, int)
train_data, train_labels = load_and_preprocess_data(train_dir, img_size)

model, history, test_data, test_labels_processed = process_labels_and_train(model, train_data, train_labels, img_size)
plot_results(history, model, test_data, test_labels_processed)

