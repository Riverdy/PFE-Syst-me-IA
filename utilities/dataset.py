# Ce module gère le dataset pour la calibration de caméra
import torch
from torch.utils.data import Dataset
import json, os
from PIL import Image
import torchvision.transforms as T

# Définition d'un dataset pour la calibration de caméra
class CameraDataset(Dataset):
    def __init__(self, annotations_path, images_dir):
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        self.images_dir = images_dir
        self.keys = list(self.annotations.keys())
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_name = self.keys[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        position = torch.tensor(self.annotations[img_name]['position'])
        rotation = torch.tensor(self.annotations[img_name]['rotation'])
        return image, position, rotation
