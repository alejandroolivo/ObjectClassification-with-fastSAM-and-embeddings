import os
from core.DetectAndClassifySigLIP import DetectAndClassifySigLIP
import numpy as np
import hashlib

DATASET = 'Fruta'

# Componer path de frames
PRODUCTOS_CLASSES_PATH = '.data/' + DATASET + '/Clases/'


# Ejemplo de uso de la clase GFTFastSAM
gft_model = DetectAndClassifySigLIP ('./models/FastSAM.pt')

def string_to_color(string):
    """Genera un color (BGR) de manera determinística a partir de un string."""
    # Usar SHA256 para generar un hash del string y tomar los primeros 3 bytes para el color
    hash_bytes = hashlib.sha256(string.encode('utf-8')).digest()
    color = (int(hash_bytes[0]), int(hash_bytes[1]), int(hash_bytes[2]))
    return color

# Cargar embbedings de cada clase
productos_classes_embeddings = gft_model.load_embeddings(PRODUCTOS_CLASSES_PATH)

# Detect classes from embeddings
productos_classes = list(productos_classes_embeddings.keys())

# Generar colores de manera determinística para cada clase
unique_classes = np.unique(productos_classes)
colors = [string_to_color(cls) for cls in unique_classes]

def detect_and_classify(Image = None, productos_classes_embeddings = None):

    boxes, ann, prompt_process = gft_model.detect_boxes_with_filters_Image(Image, imgsz=640, 
                                                                     conf=0.6, iou=0.9, 
                                                                     confidence_threshold=0.6,
                                                                     min_height=50, max_height=1800,
                                                                     min_width=100, max_width=1280,
                                                                     min_aspect_ratio=0.25, max_aspect_ratio=2.6,
                                                                     min_y1=0, max_y1=2600)
       
    # Calcular embeddings para cada recorte
    cropped_images = [gft_model.crop_image_Image(Image, bbox) for bbox in boxes]

    # Clasificar cada recorte
    productos_class_labels, productos_class_indices = gft_model.classify_crops_with_knn(cropped_images, productos_classes_embeddings, k=5)

    # Mostrar la imagen con las bounding boxes y los labels de las clases
    # gft_model.show_boxes_with_classes(Image, boxes, productos_class_indices, productos_class_labels, labels_size=10)
    
    # Filtrar las bounding boxes y los labels de las clases
    hide_classes = ['Bg']
    filtered_boxes, filtered_class_indices, filtered_class_labels = gft_model.filter_boxes_by_class(boxes, productos_class_indices, productos_class_labels, hide_classes)

    # Mostrar la imagen con las bounding boxes y los labels de las clases
    ImageOut = gft_model.show_boxes_with_classes_Image(Image, filtered_boxes, filtered_class_indices, filtered_class_labels, unique_classes, colors)

    return ImageOut

# Cargar Video
import cv2
import numpy as np

# Cargar Video
cap = cv2.VideoCapture('./frames/videos/fruit.mp4')

# Definir los FPS a los que quieres capturar y mostrar el video
fps_deseados = 5  # Ajusta este valor al deseado

# Calcular el número de fotogramas a saltar basado en los FPS originales y deseados
fps_original = cap.get(cv2.CAP_PROP_FPS)
saltar_fotogramas = int(fps_original / fps_deseados)

contador_fotogramas = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Solo procesar el fotograma si cumple con el intervalo de salto de fotogramas
        if contador_fotogramas % saltar_fotogramas == 0:
            # Suponiendo que detect_and_classify es una función que ya tienes definida
            frame_procesado = detect_and_classify(Image=frame, productos_classes_embeddings=productos_classes_embeddings)

            # Redimensionar y procesar el frame aquí
            # frame_reducido = cv2.resize(frame_procesado, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            cv2.imshow('Frame Reducido', frame_procesado)

            # Espera para mantener la tasa de FPS deseada, ajustando dinámicamente
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        contador_fotogramas += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()