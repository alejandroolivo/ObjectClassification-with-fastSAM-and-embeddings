from FastSAM.fastsam import FastSAM, FastSAMPrompt
import cv2
import os
import clip
import torch
from torch.nn.functional import cosine_similarity
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from scipy.stats import mode

class DetectAndClassifyCLIP:
    def __init__(self, model_path):
        # Cargar el modelo de FastSAM
        self.model = FastSAM(model_path)
        self.device = 'cuda:0'

        # Cargar el modelo y el preprocesamiento de CLIP
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    # DETECTION METHODS

    def detect_boxes(self, image_path, retina_masks=False, imgsz=512, conf=0.5, iou=0.9, confidence_threshold=0.6):
        # Procesar la imagen con el modelo
        everything_results = self.model(image_path, device=self.device, retina_masks=retina_masks, imgsz=imgsz, conf=conf, iou=iou)
        prompt_process = FastSAMPrompt(image_path, everything_results, device=self.device)
        
        # Obtener anotaciones y formatear resultados
        ann = prompt_process.everything_prompt()
        formatted_results = prompt_process._format_results(everything_results[0])
        
        # Extraer bounding boxes y filtrar por confianza
        boxes = []
        for annotation in formatted_results:
            bbox = annotation['bbox'].tolist()
            confidence = bbox[4]
            if confidence >= confidence_threshold:
                # Convertir a enteros y crear una lista o tupla
                bbox = [int(coordinate) for coordinate in bbox[:4]]
                boxes.append(bbox)

        return boxes, ann, prompt_process
    
    def detect_boxes_with_filters(self, image_path, retina_masks=False, imgsz=512, conf=0.5, iou=0.9, confidence_threshold=0.6, 
                                  min_height=None, max_height=None, min_width=None, max_width=None, 
                                  min_aspect_ratio=None, max_aspect_ratio=None, min_y1=None, max_y1=None):
        # Procesar la imagen con el modelo
        everything_results = self.model(image_path, device=self.device, retina_masks=retina_masks, imgsz=imgsz, conf=conf, iou=iou)
        prompt_process = FastSAMPrompt(image_path, everything_results, device=self.device)
        
        # Obtener anotaciones y formatear resultados
        ann = prompt_process.everything_prompt()
        formatted_results = prompt_process._format_results(everything_results[0])
        
        # Extraer bounding boxes y aplicar filtros
         # Extraer bounding boxes y filtrar por confianza
        boxes = []
        for annotation in formatted_results:
            bbox = annotation['bbox'].tolist()
            confidence = bbox[4]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0

            # Aplicar filtros
            if (confidence >= confidence_threshold and
                (min_height is None or height >= min_height) and
                (max_height is None or height <= max_height) and
                (min_width is None or width >= min_width) and
                (max_width is None or width <= max_width) and
                (min_aspect_ratio is None or aspect_ratio >= min_aspect_ratio) and
                (max_aspect_ratio is None or aspect_ratio <= max_aspect_ratio) and
                (min_y1 is None or y1 >= min_y1) and
                (max_y1 is None or y1 <= max_y1)):
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return boxes, ann, prompt_process

    def detect_boxes_with_filters_Image(self, image, retina_masks=False, imgsz=512, conf=0.5, iou=0.9, confidence_threshold=0.6, 
                                  min_height=None, max_height=None, min_width=None, max_width=None, 
                                  min_aspect_ratio=None, max_aspect_ratio=None, min_y1=None, max_y1=None):
        # Procesar la imagen con el modelo
        everything_results = self.model(image, device=self.device, retina_masks=retina_masks, imgsz=imgsz, conf=conf, iou=iou)
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        
        # Obtener anotaciones y formatear resultados
        ann = prompt_process.everything_prompt()
        formatted_results = prompt_process._format_results(everything_results[0])
        
        # Extraer bounding boxes y aplicar filtros
         # Extraer bounding boxes y filtrar por confianza
        boxes = []
        for annotation in formatted_results:
            bbox = annotation['bbox'].tolist()
            confidence = bbox[4]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0

            # Aplicar filtros
            if (confidence >= confidence_threshold and
                (min_height is None or height >= min_height) and
                (max_height is None or height <= max_height) and
                (min_width is None or width >= min_width) and
                (max_width is None or width <= max_width) and
                (min_aspect_ratio is None or aspect_ratio >= min_aspect_ratio) and
                (max_aspect_ratio is None or aspect_ratio <= max_aspect_ratio) and
                (min_y1 is None or y1 >= min_y1) and
                (max_y1 is None or y1 <= max_y1)):
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return boxes, ann, prompt_process

    def save_annotated_image(self, prompt_process, annotations, output_path):
        prompt_process.plot(annotations=annotations, output_path=output_path)

    def save_cropped_images(self, output_dir, bounding_boxes, image_path):
        image = cv2.imread(image_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox
            cropped_image = image[y1:y2, x1:x2]
            cropped_image_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_crop{i}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)

    def crop_image(self, image_path, bbox):
        # Leer la imagen
        image = cv2.imread(image_path)

        # Recortar la imagen según el bounding box
        x1, y1, x2, y2 = bbox
        cropped_image = image[y1:y2, x1:x2]

        return cropped_image
    
    def crop_image_Image(self, image, bbox):
        
        # Recortar la imagen según el bounding box
        x1, y1, x2, y2 = bbox
        cropped_image = image[y1:y2, x1:x2]

        return cropped_image

    def show_boxes(self, image_path, bounding_boxes):
        # Leer la imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crear figura y eje en matplotlib
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Dibujar cada bounding box en la imagen
        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"Id.:{i}", color='r')

        # Mostrar la imagen con las bounding boxes
        plt.show()

    # CLASSIFICATION METHODS

    def calculate_and_save_embeddings(self, classes_path):

        class_folders = [f for f in os.listdir(classes_path) if not f.startswith('.') and os.path.isdir(os.path.join(classes_path, f))]
        for class_folder in class_folders:
            class_path = os.path.join(classes_path, class_folder)
            embeddings_folder = os.path.join(classes_path, class_folder)
            os.makedirs(embeddings_folder, exist_ok=True)

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.png', '.bmp', '.tif', '.tiff')):
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path)

                    # Calcular el embedding
                    img_preprocessed = self.preprocess(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        img_embedding = self.clip_model.encode_image(img_preprocessed).cpu().numpy().astype(float)

                    # Guardar el embedding
                    embedding_file_path = os.path.join(embeddings_folder, img_file.rsplit('.', 1)[0] + '.npy')
                    np.save(embedding_file_path, img_embedding)

        print("Embeddings calculados y guardados con éxito.")

    def load_embeddings(self, embeddings_path):
        embeddings = {}
        class_folders = [f for f in os.listdir(embeddings_path) if not f.startswith('.') and os.path.isdir(os.path.join(embeddings_path, f))]
        for class_folder in class_folders:
            class_path = os.path.join(embeddings_path, class_folder)
            embeddings[class_folder] = []
            for embedding_file in os.listdir(class_path):
                if embedding_file.endswith('.npy'):
                    embedding_path = os.path.join(class_path, embedding_file)
                    embedding = np.load(embedding_path)
                    embeddings[class_folder].append(embedding)

        print("Embeddings cargados con éxito.")

        return embeddings  

    def compute_bboxes_embeddings(self, cropped_images):

        # Si no hay imágenes recortadas, retornar un tensor vacío
        if len(cropped_images) == 0:
            return torch.Tensor([]).to(self.device)

        # Preprocesar y calcular los embeddings para cada imagen recortada
        preprocessed_images = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).to(self.device) for crop in cropped_images])
        with torch.no_grad():
            embeddings = self.clip_model.encode_image(preprocessed_images)
        return embeddings

    def compute_bboxes_embeddings_optimized(self, cropped_images):

        if not cropped_images:
            return torch.Tensor([]).to(self.device)
        
        # Convertir y preprocesar en batch
        preprocessed_images = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            for crop in cropped_images
        ]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.clip_model.encode_image(preprocessed_images)
        
        return embeddings
    
    def classify_crops(self, crops, embeddings, mode='avg'):        
        
        crops_embeddings = self.compute_bboxes_embeddings(crops)

        # Mapear nombres de clases a índices numéricos
        class_indices = {name: idx for idx, name in enumerate(embeddings.keys())}

        # Calcular la similitud coseno para cada crop y clasificar
        class_labels = []
        class_indices_list = []
        for img_embedding in crops_embeddings:
            similarities = {}
            for class_name, class_embeddings in embeddings.items():
                cos_scores_total = 0
                max_value = 0
                for class_embedding in class_embeddings:
                    class_embedding_tensor = torch.Tensor(class_embedding).to(self.device)
                    cos_scores = cosine_similarity(img_embedding, class_embedding_tensor)
                    cos_scores_total += cos_scores
                    max_value = max(max_value, cos_scores)
                if mode == 'max':
                    similarities[class_name] = max_value
                elif mode == 'avg':
                    similarities[class_name] = cos_scores_total / len(class_embeddings)
    
            # Obtener la clase con la mayor similitud
            best_class = max(similarities, key=similarities.get)
            class_labels.append(best_class)
            class_indices_list.append(class_indices[best_class])

        return class_labels, class_indices_list
    
    def classify_crops_with_knn(self, crops, embeddings, k=5):

        crops_embeddings = self.compute_bboxes_embeddings_optimized(crops).to('cpu').numpy()
        class_indices = {name: idx for idx, name in enumerate(embeddings.keys())}
        
        # Convertir embeddings de todas las clases a una lista de arrays y crear etiquetas
        all_class_embeddings = []
        all_class_labels = []
        for class_name, class_embeddings in embeddings.items():
            for embedding in class_embeddings:
                all_class_embeddings.append(np.array(embedding))
                all_class_labels.append(class_indices[class_name])

        all_class_embeddings_array = np.stack(all_class_embeddings)
        all_class_labels_array = np.array(all_class_labels)
        
        class_labels = []
        class_indices_list = []
        # Suponiendo que crops_embeddings es un array de forma (n_crops, 512)
        for i, img_embedding in enumerate(crops_embeddings):
            # Calcular las distancias para el i-ésimo crop
            distances = np.linalg.norm(all_class_embeddings_array.squeeze() - img_embedding, axis=1)
            # Encontrar los k índices de las menores distancias para este crop
            knn_indices = np.argsort(distances)[:k]
            # Obtener las etiquetas de los k vecinos más cercanos para este crop
            knn_labels = all_class_labels_array[knn_indices]
            # Resto del código para obtener la etiqueta más común, etc.
            # Encontrar la etiqueta más común entre los k vecinos más cercanos
            most_common_label = mode(knn_labels).mode[0]
            class_label = list(class_indices.keys())[list(class_indices.values()).index(most_common_label)]
            class_labels.append(class_label)
            class_indices_list.append(most_common_label)
        
        return class_labels, class_indices_list

    def filter_boxes_by_class(self, boxes, class_indices, class_labels, hide_classes):
        # Filtrar las boxes, class_indices y class_labels para excluir las clases en hide_classes
        filtered_boxes = []
        filtered_class_indices = []
        filtered_class_labels = []

        for box, class_index, class_label in zip(boxes, class_indices, class_labels):
            if class_label not in hide_classes:
                filtered_boxes.append(box)
                filtered_class_indices.append(class_index)
                filtered_class_labels.append(class_label)

        return filtered_boxes, filtered_class_indices, filtered_class_labels

    def compute_blurred_bboxes_embeddings(self, cropped_images, k_size=5):

        # Preprocesar y calcular los embeddings para cada imagen recortada
        # Aplicar un filtro de desenfoque a las imágenes recortadas y luego preprocesarlas
        preprocessed_images = []
        for crop in cropped_images:
            # Aplicar el filtro de desenfoque
            blurred_crop = cv2.GaussianBlur(crop, (k_size, k_size), 0)

            # Convertir a formato PIL y preprocesar
            pil_image = Image.fromarray(cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2RGB))
            preprocessed_image = self.preprocess(pil_image).to(self.device)
            preprocessed_images.append(preprocessed_image)

        preprocessed_images = torch.stack(preprocessed_images)

        with torch.no_grad():
            embeddings = self.clip_model.encode_image(preprocessed_images)
        return embeddings

    # CLUSTERING METHODS

    def compute_additional_features(self, bounding_boxes):
        # Calcula características adicionales como 'y1' y área de la bbox
        features = []
        for bbox in bounding_boxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height != 0 else 0
            features.append([y1, area, aspect_ratio])
        return np.array(features)

    def estimate_optimal_clusters(self, embeddings, max_clusters=10):
        # Método del Codo para estimar el número óptimo de clusters
        sum_of_squared_distances = []
        for k in range(1, max_clusters):
            km = KMeans(n_clusters=k)
            km = km.fit(embeddings)
            sum_of_squared_distances.append(km.inertia_)

        # Puedes graficar aquí la curva del codo para visualizar y elegir el número óptimo de clusters

        # Aquí simplemente retornamos el número de clusters que minimiza la suma de distancias cuadradas
        return np.argmin(sum_of_squared_distances) + 1

    def perform_clustering_prev(self, embeddings, n_clusters=5):
        # Realizar clustering sobre los embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings.cpu())
        return kmeans.labels_
    
    def perform_clustering(self, embeddings, additional_features, n_clusters=None):
        # Combinar embeddings con características adicionales
        if additional_features is not None:
            combined_features = np.hstack((embeddings.cpu().numpy(), additional_features))
        else:
            combined_features = embeddings.cpu().numpy()

        # Estimar el número óptimo de clusters si no se proporciona
        if n_clusters is None:
            n_clusters = self.estimate_optimal_clusters(combined_features)

        # Realizar clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_features)
        return kmeans.labels_

    def perform_clustering_from_features(self, additional_features, n_clusters=None):
        # Estimar el número óptimo de clusters si no se proporciona
        if n_clusters is None:
            n_clusters = self.estimate_optimal_clusters(additional_features)

        # Realizar clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(additional_features)
        return kmeans.labels_

    def show_boxes_with_classes(self, image_path, bounding_boxes, classes, labels, labels_size=10):
        # Leer la imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crear figura y eje en matplotlib
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Colores para cada clase
        colors = plt.cm.get_cmap('tab10', len(np.unique(classes)))

        # Dibujar cada bounding box en la imagen
        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            cls = classes[i]
            lbl = labels[i]
            color = colors(cls)

            # Añadir color de relleno con transparencia
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor=color, alpha=0.2)
            ax.add_patch(rect)
            ax.text(x1, y1, f"Cls:{lbl}", color=color, fontsize=labels_size, rotation=30, weight='bold')

        # Mostrar la imagen con las bounding boxes y clases
        plt.show()

    def show_boxes_with_classes_Image(self, image, bounding_boxes, classes, labels, unique_classes, colors):
        # Convertir la imagen a RGB si está en BGR (OpenCV usa BGR por defecto)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        class_to_color = dict(zip(unique_classes, colors))

        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            cls = classes[i]
            lbl = labels[i]
            color = class_to_color[lbl]

            # Dibujar el rectángulo de la bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 3)

            # Preparar el texto a dibujar
            text = f"Cls:{lbl}"
            
            # Calcular el tamaño del texto para posicionarlo correctamente
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0] * 2

            # Dibujar un fondo para el texto
            cv2.rectangle(image_rgb, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            
            # Dibujar el texto
            cv2.putText(image_rgb, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # Convertir la imagen de vuelta a BGR para mostrarla con OpenCV (opcional)
        image_bgr_with_boxes = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return image_bgr_with_boxes

