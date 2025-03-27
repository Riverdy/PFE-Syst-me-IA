# Ce module évalue après entrainement le modèle de calibration de caméra
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.calibration_model import CameraCalibrationNet
from utilities.dataset import CameraDataset

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, positions, rotations in dataloader:
            images, positions, rotations = images.cuda(), positions.cuda(), rotations.cuda()
            pred_positions, pred_rotations = model(images)

            # Calcul de la perte pour la position (SmoothL1Loss)
            loss_position = criterion(pred_positions, positions)
            loss_rotation = criterion(pred_rotations, rotations)

            # Perte totale
            loss = loss_position + loss_rotation
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Average Loss on test set: {avg_loss}')

if __name__ == '__main__':
    test_dataset = CameraDataset('datas/annotations.json', 'datas/images')
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Chargement du modèle entraîné
    model = CameraCalibrationNet().cuda()
    model.load_state_dict(torch.load('outputs/model.pth'))

    # Définition des fonctions de perte
    criterion = torch.nn.SmoothL1Loss()

    # Exécution de l'évaluation
    evaluate(model, test_loader, criterion)

