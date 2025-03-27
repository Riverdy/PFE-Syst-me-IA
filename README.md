Conception d'un système IA entrainé pour l'extraction de données de calibration de caméra à partir de photos autour d'un modèle 3D.

Executer le programme lance un entrainement sur une base de 5000 images pour un reseau de neurones convolutifs avec une fonction de perte Huber Loss.
Des prédictions sont ensuite réalisées sur 100 images test.

!! Vous devez charger vous meme vos données d'entrainements dans des dossiers images dans les dossiers datas et toPredict ainsi que leur annotations associés !!
Nous proposons un script Blender pour génerer synthétiquement le nombre d'images souhaités ainsi que les annotations correspondantes dans un fichier JSON.
