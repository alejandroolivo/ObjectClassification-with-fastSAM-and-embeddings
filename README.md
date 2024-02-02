# Clasificación de Objetos con FastSAM y Embeddings

## Descripción General

Este proyecto presenta una solución de Computer Vision para la detección y clasificación de objetos en imágenes, las cuales son extraídas como frames de vídeos. Utiliza el modelo FastSAM para la detección de objetos, y para la clasificación, emplea embeddings que pueden ser generados mediante dos modelos distintos: CLIP o SigLIP. La comparación de embeddings se realiza a través de la similitud coseno o mediante el algoritmo KNN, proporcionando una estructura flexible y robusta para abordar diversos desafíos de clasificación en el campo de la visión por computadora.

El proyecto está estructurado en directorios específicos para mantener los datos, los modelos, la información de clasificación y los resultados de salida, asegurando así una organización y un flujo de trabajo claros. La carpeta `FastSAM` es esencial para el funcionamiento del modelo de detección, mientras que la carpeta `models` alberga los modelos de clasificación de embeddings necesarios.

## Preparación

### Descarga del Repositorio de FastSAM

Además de los pesos, es necesario descargar el código fuente de FastSAM desde su repositorio oficial:

git clone https://github.com/CASIA-IVA-Lab/FastSAM

### Descarga de Pesos del Modelo FastSAM

Para usar el modelo FastSAM, es necesario descargar los pesos preentrenados. Sigue el siguiente enlace para descargar el archivo de pesos `FastSAM.pt` y colócalo dentro de la carpeta `models` de tu proyecto.

[Descargar pesos FastSAM](https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt)

### Instalación

Para preparar el entorno necesario para ejecutar este proyecto, sigue los siguientes pasos:

1. Clona el repositorio en tu máquina local:

```
git clone https://github.com/alejandroolivo/ObjectClassification-with-fastSAM-and-embeddings.git
```

2. Navega al directorio del proyecto:
```
cd OBJECTCLASSIFICATION-WITH-FASTSAM-AND-EMBEDDINGS
```

(Opcional) Crea y activa un entorno virtual:
```
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```
3. Instala las dependencias necesarias:
```
pip install -r requirements.txt
```
Con estos pasos, tu entorno debería estar listo para ejecutar los scripts y utilizar el modelo FastSAM junto con las funcionalidades de clasificación mediante embeddings.

## Estructura del Proyecto

El proyecto tiene la siguiente estructura de carpetas y archivos:

OBJECTCLASSIFICATION-WITH-FASTSAM-AND-EMBEDDINGS/
│
├── core/ # Módulos principales del proyecto
│ ├── DetectAndClassifyCLIP.py # Clase para la detección y clasificación con CLIP
│ └── DetectAndClassifySigLIP.py # Clase para la detección y clasificación con SigLIP
│
├── data/ # Datos utilizados para ejemplos y pruebas
│
├── examples/ # Scripts de ejemplo para demostrar el uso de los módulos core
│ ├── ClipAndMLPTraining.py
│ ├── CreateEmbeddings_ObjectClassificationCLIP.py
│ ├── CreateEmbeddings_ObjectClassificationSigLIP.py
│ ├── FastSAMFullExample.py
│ ├── FewShotObjectClassificationCLIP_VIDEO.py
│ ├── FewShotObjectClassificationSigLIP_VIDEO.py
│ ├── SigLIPEmbeddingsVisualization.py
│ └── ZeroShotObjectClassification.py
│
├── FastSAM/ # Código fuente y dependencias para el modelo FastSAM
│
├── info/ # Documentación e información relevante
│
├── models/ # Modelos entrenados, incluyendo pesos de FastSAM
│
├── output/ # Salida generada por los scripts, como clasificaciones y visualizaciones
│
├── utils/ # Herramientas de utilidad para operaciones generales como preprocesamiento
│ ├── CropSavingWithFastSAM.py
│ └── CustomDataset.py
