# Ce module entraîne le modèle de calibration de caméra
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.calibration_model import CameraCalibrationNet
from utilities.dataset import CameraDataset
from torch.optim.lr_scheduler import StepLR

# Récupération du dataset
dataset = CameraDataset('datas/annotations.json', 'datas/images')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Création du modèle de calibration de caméra sur le GPU
model = CameraCalibrationNet().cuda()
# La fonction de perte (ici SLL)
criterion = torch.nn.SmoothL1Loss()

# Fonction d'optimisation Adam avec un learning rate au départ de 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Ajout du scheduler pour réduire le learning rate après un certain nombre d'epochs
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)  # Réduit le LR par un facteur 0.1 tous les 6 epochs

# Le modèle est entraîné sur 20 epochs
for epoch in range(20):
    # Mode entraînement
    model.train()
    for images, positions, rotations in loader:
        images, positions, rotations = images.cuda(), positions.cuda(), rotations.cuda()
        # Prédictions
        pred_positions, pred_rotations = model(images)
        # Perte pour la position
        loss_position = criterion(pred_positions, positions)
        # Perte pour la rotation
        loss_rotation = criterion(pred_rotations, rotations)
        # Perte totale
        loss = loss_position + loss_rotation 
        # Réinitialisation des gradients
        optimizer.zero_grad()
        # Retropropagation (calcul des gradients)
        loss.backward()
        # Mise à jour des poids
        optimizer.step()
    
    # Mise à jour du learning rate selon le scheduler
    scheduler.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

# Sauvegarde du modele entrainé
torch.save(model.state_dict(), 'outputs/model.pth')

