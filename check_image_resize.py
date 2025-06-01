#type: ignore

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
import numpy as np

# Load an actual image (you can use your own file path)
image_path = "/persistent/home/anik/Research/Jujube Leaf/Coding Files/boroi_pata/healty/IMG_20250318_131049.jpg"
original_img = load_img(image_path)  # Loads image in PIL format
original_array = img_to_array(original_img)  # Convert to NumPy array

# Resize to MobileNetV2 size
resized_array = resize(original_array, [224, 224]).numpy().astype("uint8")

# Plot both images
plt.figure(figsize=(10, 5))

# Original
plt.subplot(1, 2, 1)
plt.imshow(original_array.astype("uint8"))
plt.title("Original Image")
plt.axis("off")

# Resized
plt.subplot(1, 2, 2)
plt.imshow(resized_array)
plt.title("Resized to 224x224")
plt.axis("off")

plt.tight_layout()
plt.show()
