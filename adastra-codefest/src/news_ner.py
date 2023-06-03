# Importación de bibliotecas
import spacy
import pickle
import pandas as pd
import json
import csv
import requests
from bs4 import BeautifulSoup
import re
from flair.data import Sentence
from flair.models import SequenceTagger

# Carga del modelo de clasificación de texto y creación de modelos de procesamiento de lenguaje en spaCy
def inicioPrograma():
  # Carga del modelo de clasificación a partir de un archivo pickle
  with open('bd.pkl', 'rb') as archivo:
    dic_cargado = pickle.load(archivo)

  # Inicialización de un nuevo modelo de procesamiento de lenguaje en spaCy en español
  nlp = spacy.blank("es")

  # Añade un categorizador de texto al pipeline del modelo de procesamiento de lenguaje,
  # utilizando un modelo preentrenado y los datos cargados anteriormente
  nlp.add_pipe(
    "text_categorizer", 
    config={
        "data": dic_cargado, 
        "model": "hiiamsid/sentence_similarity_spanish_es",
        "device": "gpu"
    }
  )

  # Inicialización de un modelo de sentencia en español en spaCy y añade un sentencizador al pipeline
  sentence_model = spacy.blank("es")
  sentence_model.add_pipe("sentencizer")

  # Devuelve los modelos de procesamiento de lenguaje y sentencia para su uso posterior
  return sentence_model, nlp

# Clasificación del texto del documento según ciertas categorías de problemas medioambientales
def classification(doc):
  # Cada categoría tiene asignada una puntuación de similitud, si supera un umbral se asume que el documento pertenece a esa categoría
  if doc._.cats['CONTAMINACION']>=0.7:
    return ("Afectación al medio ambiente que puede ser CONTAMINACION")
  elif doc._.cats['DEFORESTACION']>=0.7:
    return ("Afectación al medio ambiente que puede ser DEFORESTACION")
  elif doc._.cats['MINERIA']>=0.7:
    return ("Afectación al medio ambiente que puede ser MINERA")
  else:
    return ("Afectación al medio ambiente que puede ser NINGUNA")

# Extracción de entidades nombradas (NER) del texto utilizando Flair
def tokens_ner(text):
    # Listas para almacenar las entidades extraídas
    entities_loc = []
    entities_per = []
    entities_org = []
    entities_misc = []

    # Carga el modelo de NER y procesa el texto
    tagger = SequenceTagger.load("flair/ner-spanish-large")
    sentence = Sentence(text)
    tagger.predict(sentence)

    # Clasifica las entidades extraídas por tipo y las añade a las listas correspondientes
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'LOC':
            entities_loc.append(entity)
        elif entity.tag == 'PER':
            entities_per.append(entity)
        elif entity.tag == 'ORG':
            entities_org.append(entity)
        elif entity.tag == 'MISC':
            entities_misc.append(entity)

    # Devuelve las listas de entidades
    return entities_org, entities_loc, entities_per, [], entities_misc

# Genera un archivo JSON con los resultados del análisis de texto
def finalizar(list1, list2, list3, list4, list5, string1, string2, path):
  # Creación de un diccionario con los resultados
  diccionario = {
    "text": string1,
    "org": str(list1),
    "loc": str(list2),
    "per": str(list3),
    "dates": str(list4),
    "misc": str(list5),
    "impact": string2
  }

  # Conversión del diccionario a JSON y escritura en el archivo de salida
  json_data = json.dumps(diccionario)
  with open(path+".json", "w") as archivo:
    archivo.write(json_data)

# Funciones que permiten analizar texto desde una cadena, un archivo o una URL
def ner_from_str(text, output_path):
  sentence_model, nlp =inicioPrograma()
  sentences = sentence_model(text)
  doc = nlp(sentences.text)
  impact= classification(doc)
  list1, list2, list3, list4, list5=tokens_ner(text)
  finalizar(list1, list2, list3, list4, list5, text, impact, output_path)

def ner_from_file(text_path, output_path):
  sentence_model, nlp =inicioPrograma()
  try:
    with open(text_path, 'r') as archivo:
        text = archivo.read()
    sentences = sentence_model(text)
    doc = nlp(sentences.text)
    impact= classification(doc)
    list1, list2, list3, list4, list5=tokens_ner(text)
    finalizar(list1, list2, list3, list4, list5, text, impact, output_path)
  except:
    try:
      with open('archivo.csv', 'r') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            text=str(fila)
      sentences = sentence_model(text)
      doc = nlp(sentences.text)
      impact= classification(doc)
      list1, list2, list3, list4, list5=tokens_ner(text)
      finalizar(list1, list2, list3, list4, list5, text, impact, output_path)
    except:
      pass

def ner_from_url(url, output_path):
  sentence_model, nlp =inicioPrograma()
  text=''
  response = requests.get(url)
  if response.status_code == 200:
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        parrafos = soup.find_all('p')
        texto_extraido = [re.sub(r'\s+', ' ', parrafo.get_text(strip=True)) for parrafo in parrafos]
        for parrafo in texto_extraido:
            text=text.join(parrafo)
            if len(text)>150:
              break
        sentences = sentence_model(text)
        doc = nlp(sentences.text)
        impact= classification(doc)
        list1, list2, list3, list4, list5=tokens_ner(text)
        finalizar(list1, list2, list3, list4, list5, text, impact, output_path)
  else:
      pass

# Ejemplo de uso de la función ner_from_url
ner_from_url("https://www.lapatilla.com/2023/05/31/la-mineria-ilegal-crece-voraz-y-amenazante-en-la-amazonia-de-ecuador/","prueba4")
