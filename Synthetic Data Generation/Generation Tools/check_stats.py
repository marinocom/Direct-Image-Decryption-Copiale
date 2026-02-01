# --------------------------
# Utils
# Check stats of the dataset: number of images and JSON entries
# --------------------------


import os
import json

datasetPath = ''

splits = ['train', 'val', 'test']

for split in splits:
    img_dir = os.path.join(datasetPath, split)
    json_path = os.path.join(datasetPath, f'{split}.json')

    # Count images
    if os.path.exists(img_dir):
        img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        num_imgs = len(img_files)
    else:
        num_imgs = 0
    # Count JSON entries
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        num_json = len(data)
    else:
        num_json = 0
    print(f"{split}: {num_imgs} images, {num_json} JSON entries") 

    # total
total_imgs = 0
total_json = 0
for split in splits:
    img_dir = os.path.join(datasetPath, split)
    json_path = os.path.join(datasetPath, f'{split}.json')

    # Count images
    if os.path.exists(img_dir):
        img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        total_imgs += len(img_files)

    # Count JSON entries
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        total_json += len(data)
print(f"Total: {total_imgs} images, {total_json} JSON entries")