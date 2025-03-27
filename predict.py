# Ce module prédit la position et la rotation d'une image donnée
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from models.calibration_model import CameraCalibrationNet
import os 

def predict_image(model, image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # Vérification de l'exxistance de l'image
    if not os.path.exists(image_path):
        print(f"❌ Fichier introuvable : {image_path}")
        return None, None

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).cuda()
    
    model.eval()
    with torch.no_grad():
        pred_position, pred_rotation = model(image_tensor)

    return pred_position.cpu().numpy()[0], pred_rotation.cpu().numpy()[0]

if __name__ == '__main__':
    # Chargement du modèle
    model = CameraCalibrationNet().cuda()
    model.load_state_dict(torch.load('outputs/model.pth'))
    
    for i in range(0, 100):
        image_path = f"toPredict/images/image_{i:04d}.png"

        # Prédictions
        position, rotation = predict_image(model, image_path)

        # Résultats
        if position is not None and rotation is not None:
            print(f"✅ Image {image_path}")
            print(f"   - Position prédite : {position}")
            print(f"   - Rotation prédite (normalisée) : {rotation}")
        else:
            print(f"❌ Échec de la prédiction pour {image_path}")


   
