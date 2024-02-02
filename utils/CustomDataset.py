import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import PIL.Image as Image

class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.classes = os.listdir(main_dir)  # List of classes by directory names
        self.images = []
        self.labels = []
        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(main_dir, class_name)
            for image_name in os.listdir(class_dir):
                if image_name.endswith('.png') and not image_name.endswith('.ipynb_checkpoints'):  # Filtrar solo imágenes .png
                    self.images.append(os.path.join(class_dir, image_name))
                    self.labels.append(class_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_class_name(self, class_idx):
        return self.classes[class_idx]

# # Define the transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
# ])

# # Load your custom dataset
# dataset = CustomImageDataset(main_dir='path_to_your_images', transform=transform)
