import os
from PIL import Image

def find_corrupted_images(directory):
    corrupted_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                path = os.path.join(root, file)
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                corrupted_files.append(path)
    return corrupted_files

# Replace this with your training folder path
dataset_path = r"D:\Research\Jujube\boroi"  
corrupted = find_corrupted_images(dataset_path)

print("Corrupted Images:")
for file in corrupted:
    print(file)
