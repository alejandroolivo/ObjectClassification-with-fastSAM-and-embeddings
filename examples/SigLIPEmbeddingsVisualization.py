import os
import cv2
from transformers import AutoProcessor, SiglipVisionModel
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Camino a la carpeta de datos
DATASET = 'Fruta'
DIMENSIONS = 3  # 2 o 3

# Componer path de clases
data_path = './data/' + DATASET + '/Clases/'

# Cargar el modelo y el procesador
model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Diccionario para guardar los embeddings y las etiquetas
embeddings = {}

# si existe un archivo embeddings.pt, lo carga 
if os.path.exists(os.path.join(data_path, 'embeddings.pt')):
    embeddings = torch.load(os.path.join(data_path, 'embeddings.pt'))
    
else:
    # si no existe, lo crea
    embeddings = {}

    # Recorrer las carpetas y subcarpetas
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = os.path.basename(root)
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Verificar si la imagen se cargó correctamente
                if image is not None:
                    # Procesar la imagen
                    inputs = processor(images=image, return_tensors="pt")

                    # Pasar la imagen procesada al modelo
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Obtener el pooled output (embedding) de la imagen
                    embedding = outputs.pooler_output.cpu().numpy()

                    # Guardar el embedding y la etiqueta
                    if class_name not in embeddings:
                        embeddings[class_name] = []
                    embeddings[class_name].append(embedding)


    # Guarda el diccionario de embeddings en un archivo en la ruta data_path
    torch.save(embeddings, os.path.join(data_path, 'embeddings.pt'))

# Asumiendo que 'embeddings' es el diccionario que has cargado
flat_embeddings = []
labels = []

for class_name, class_embeddings in embeddings.items():
    for embedding in class_embeddings:
        flat_embeddings.append(embedding)
        labels.append(class_name)

flat_embeddings = np.array(flat_embeddings).squeeze()

if DIMENSIONS == 2:

    # Configuración de t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=3500)
    tsne_results = tsne.fit_transform(flat_embeddings)

    import matplotlib.pyplot as plt

    # Crear el gráfico
    plt.figure(figsize=(16, 10))

    # Colores únicos para cada clase
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Dibujar cada punto
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors[i], label=label)

    # Añadir leyenda y etiquetas
    plt.legend()
    plt.title('Visualización de Embeddings con t-SNE')
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')

    # Mostrar el gráfico
    plt.show()

elif DIMENSIONS == 3:

    # Configuración de t-SNE
    tsne = TSNE(n_components=3, verbose=1, perplexity=10, n_iter=4500)
    tsne_results = tsne.fit_transform(flat_embeddings)

    # Crear una figura para el gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Colores únicos para cada clase
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Asignar un color a cada punto basado en su etiqueta
    point_colors = [color_map[label] for label in labels]

    # Dibujar cada punto en el espacio 3D
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], color=point_colors)

    # Añadir leyenda (creando un objeto de leyenda manualmente)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color_map[label], markersize=10) for label in unique_labels]
    ax.legend(handles=legend_elements)

    ax.set_title('Visualización de Embeddings 3D con t-SNE')
    ax.set_xlabel('Componente t-SNE 1')
    ax.set_ylabel('Componente t-SNE 2')
    ax.set_zlabel('Componente t-SNE 3')

    # Mostrar el gráfico
    plt.show()