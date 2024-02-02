import os
from core.DetectAndClassifySigLIP import DetectAndClassifySigLIP

DATASET = 'Fruta'

# Componer path de frames
CLASSES_PATH = '.data/' + DATASET + '/Clases/'

# Calcular embeddings para cada clase
classes = [f for f in os.listdir(CLASSES_PATH) if os.path.isfile(os.path.join(CLASSES_PATH, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Ejemplo de uso
gft_model = DetectAndClassifySigLIP ('./models/FastSAM.pt')

gft_model.calculate_and_save_embeddings(CLASSES_PATH)