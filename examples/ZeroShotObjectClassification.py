import os
from core.DetectAndClassifyCLIP import DetectAndClassifyCLIP

DATASET = 'Fruta'

# Componer path de frames
FOLDER_PATH = '.data/' + DATASET + '/Frames/'

# Listar todos los archivos de imagen en la carpeta
image_files = [f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Ejemplo de uso de la clase DetectAndClassifyCLIP
gft_model = DetectAndClassifyCLIP('./models/FastSAM.pt')

# Procesar cada imagen en la carpeta
for image_file in image_files:
    image_path = os.path.join(FOLDER_PATH, image_file)
    print(f"Procesando: {image_file}")

    # Detectar bounding boxes en la imagen
    boxes, ann, prompt_process = gft_model.detect_boxes_with_filters(image_path, imgsz=640, 
                                                                     conf=0.6, iou=0.9, 
                                                                     confidence_threshold=0.6,
                                                                     min_height=50, max_height=1800,
                                                                     min_width=100, max_width=1280,
                                                                     min_aspect_ratio=0.25, max_aspect_ratio=2.6,
                                                                     min_y1=0, max_y1=2600)
    
    # Calcular embeddings para cada recorte
    cropped_images = [gft_model.crop_image(image_path, bbox) for bbox in boxes] # Asumiendo que ya tienes un método 'crop_image'
    embeddings = gft_model.compute_bboxes_embeddings(cropped_images)

    # Calcular características adicionales
    additional_features = gft_model.compute_additional_features(boxes)

    # Realizar clustering sobre los embeddings
    classes = gft_model.perform_clustering(embeddings, additional_features)
    # print(classes)

    # Mostrar la imagen con las bounding boxes y sus clases
    gft_model.show_boxes_with_classes(image_path, boxes, classes, classes)