import cv2
import numpy as np
import os

def enhance_image(image):
    # Convert to LAB color space to apply CLAHE on the L-channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to reduce highlights/shadows
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge and convert back
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, file_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            enhanced = enhance_image(image)

            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, enhanced)

    print("Processing complete. Output saved in:", output_folder)

# Example usage
input_folder = "D:\Research\Drinks Classification\Image"      # Replace with your folder name
output_folder = "D:\Research\Drinks Classification\Image\output_corrected" # Folder to save results
process_images(input_folder, output_folder)
