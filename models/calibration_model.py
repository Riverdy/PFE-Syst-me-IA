# Ce module définit un réseau de neurones pour la calibration de caméra
import torch.nn as nn
import torchvision.models as models

class CameraCalibrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Utilisation de ResNet18 comme backbone, pré-entraîné sur ImageNet
        backbone = models.resnet18(pretrained=True)
        # Remplacement de la couche de classification par une couche d'identité
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # Définition de la couche de prédiction de la position
        self.fc_position = nn.Linear(512, 3)
        # Définition de la couche de prédiction de la rotation
        self.fc_rotation = nn.Linear(512, 3)

    # Méthode de forward pour le réseau de neurones
    def forward(self, x):
        # Extraction des features à l'aide du backbone
        features = self.backbone(x)
        # Prédiction de la position
        position = self.fc_position(features)
        # Prédiction de la rotation
        rotation = self.fc_rotation(features)
        return position, rotation


