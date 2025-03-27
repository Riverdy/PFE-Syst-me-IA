import torch
import torch.nn as nn
import torchvision.models as models
import timm #Pour le transformers

class CameraCalibrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN (ResNet18) pour extraire les features
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.cnn = backbone
        
        # Transformer Encoder pour traiter les features 1D
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Couches de sortie pour prédire position et rotation
        self.fc_position = nn.Linear(512, 3)
        self.fc_rotation = nn.Linear(512, 3)

    def forward(self, x):
        # Extraction des features avec ResNet18
        cnn_features = self.cnn(x)  # (batch_size, 512)
        
        # Transformer attend
        trans_input = cnn_features.unsqueeze(0)

        # Passage dans le Transformer Encoder
        trans_features = self.transformer_encoder(trans_input)
        trans_features = trans_features.squeeze(0)

        # Prédictions finales
        position = self.fc_position(trans_features)
        rotation = self.fc_rotation(trans_features)
        
        return position, rotation

# Test du modèle
if __name__ == "__main__":
    model = CameraCalibrationNet().cuda()
    dummy_input = torch.randn(2, 3, 224, 224).cuda()
    pos, rot = model(dummy_input)
    print("Position:", pos.shape, "Rotation:", rot.shape)



