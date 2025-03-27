import bpy
import math
import mathutils
import random
import os
import json

# ====================== INITIALISATION ======================

model_path = "//dog.glb"
output_dir = "//images/"
output_file = "//annotations.json"

num_cameras = 100 # Nombre de screens (nbre de positions de caméras)
min_radius = 7  # Rayon min de la sphère de positionnement des caméras autour de l'objet
max_radius = 9  # Rayon max de la sphère de positionnement des caméras autour de l'objet
object_center = mathutils.Vector((0, 0, 1.5))
min_distance = 0.05  # Distance minimale entre les caméras

# Cleaning de la scène
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ====================== IMPORT DU MODÈLE 3D ======================

abs_model_path = bpy.path.abspath(model_path)
if not os.path.exists(abs_model_path):
    raise FileNotFoundError(f"Modèle non trouvé : {abs_model_path}")

bpy.ops.import_scene.gltf(filepath=abs_model_path)

# On met le modèle au centre de la scène
imported_objects = [obj for obj in bpy.context.selected_objects]
if not imported_objects:
    raise RuntimeError("Aucun objet importé.")
model = imported_objects[0]
model.location = object_center
model.name = "TargetModel"
model.scale *= 1

# Augmente la lumière de l'environnement
bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 8.0

# ====================== CREATION DE LA CAMERA ======================

bpy.ops.object.camera_add()
cam = bpy.context.object
cam.name = "MainCamera"

# Stocker les positions des caméras
camera_positions = {}

# Génération des positions
def random_point_on_sphere(min_radius, max_radius):
    """ Génère un point aléatoire dans un rayon compris entre min_radius et max_radius """
    radius = random.uniform(min_radius, max_radius)
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(2 * random.uniform(0, 1) - 1)
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    return mathutils.Vector((x, y, z))

def is_too_close(new_pos, positions, min_dist):
    """ Vérifie que le nouveau point est à au moins 'min_dist' de tous les autres """
    for pos in positions:
        if (new_pos - pos).length < min_dist:
            return True
    return False

# Génération des prises de vue
positions_list = []
while len(positions_list) < num_cameras:
    new_pos = random_point_on_sphere(min_radius, max_radius)
    if not is_too_close(new_pos, positions_list, min_distance):
        positions_list.append(new_pos)

for i, pos in enumerate(positions_list, start=0):
    # Déplacer la caméra
    cam.location = pos
    direction = object_center - cam.location
    cam.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    cam.rotation_euler.rotate_axis('X', math.pi)
    cam.rotation_euler.rotate_axis('Z', math.pi)

    # Enregistrer la position + rotation de la caméra
    filename = f"image_{i:04d}.png"
    camera_positions[filename] = {
        "position": [round(cam.location.x, 3), round(cam.location.y, 3), round(cam.location.z, 3)],
        "rotation": [round(cam.rotation_euler.x, 3), round(cam.rotation_euler.y, 3), round(cam.rotation_euler.z, 3)]
    }

    # Prise du screen
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = bpy.path.abspath(output_dir + filename)
    bpy.ops.render.render(write_still=True)

# Sauvegarde du fichier JSON
abs_output_file = bpy.path.abspath(output_file)
with open(abs_output_file, "w") as f:
    json.dump(camera_positions, f, indent=2)

print(f"Les positions des caméras ont été enregistrées dans {output_file}")
print(f"Les rendus sont enregistrés dans {output_dir}")
