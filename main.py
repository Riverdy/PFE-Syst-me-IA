# Ce fichier exécute les étapes principales du projet
# Il utilise des scripts pour entraîner, évaluer et prédire

# Importation du module subprocess pour exécuter des commandes système
import subprocess
import time

if __name__ == '__main__':
    start_time = time.time() 
    print("Entrainement du modèle...")
    subprocess.run(['python', 'train.py'], check=True)

    print("Évaluation du modèle...")
    subprocess.run(['python', 'evaluate.py'], check=True)

    print("Prédiction exemple...")
    subprocess.run(['python', 'predict.py'], check=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time  #temps écoulé

    print(f"Prédiction terminée. Temps total : {elapsed_time:.2f} secondes.")