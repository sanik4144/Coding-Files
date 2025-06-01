import os
from PIL import Image

def crop_non_square(image_path, output_path):
    """Crops an image to a square only if it's not already square."""
    with Image.open(image_path) as img:
        if img.width == img.height:
            # Already square, save without modification
            img.convert("RGB").save(output_path, "JPEG", quality=95, optimize=True)
            return

        # Find the maximum possible square crop
        new_size = min(img.width, img.height)
        left = (img.width - new_size) // 2
        top = (img.height - new_size) // 2
        right = left + new_size
        bottom = top + new_size

        # Crop the image
        img_cropped = img.crop((left, top, right, bottom)).convert("RGB")

        # Extract EXIF data (if exists)
        exif_data = img.info.get("exif")

        # Save the cropped image as JPEG (skip exif if None)
        if exif_data:
            img_cropped.save(output_path, "JPEG", quality=95, optimize=True, exif=exif_data)
        else:
            img_cropped.save(output_path, "JPEG", quality=95, optimize=True)

        print(f"Cropped: {os.path.basename(image_path)}")

def process_images(input_folder):
    """Processes all images in the folder and subfolders, cropping only non-square ones."""
    for subdir, _, files in os.walk(input_folder):
        output_subdir = subdir + "_cropped"
        os.makedirs(output_subdir, exist_ok=True)

        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(subdir, filename)
                output_path = os.path.join(output_subdir, os.path.splitext(filename)[0] + ".jpg")
                crop_non_square(input_path, output_path)

    print("Processing complete!")

# Define folder path
input_folder = r"/persistent/home/anik/Research/Jujube Leaf/Coding Files/boroi_pata/yellow"
process_images(input_folder)













# import os
# from PIL import Image

# def count_non_square_images(folder_path):
#     non_square_count = 0

#     for subdir, _, files in os.walk(folder_path):
#         for filename in files:
#             if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#                 image_path = os.path.join(subdir, filename)
#                 with Image.open(image_path) as img:
#                     if img.width != img.height:
#                         non_square_count += 1
#                         print(f"Non-square: {filename} ({img.width}x{img.height})")

#     print(f"\nTotal non-square images: {non_square_count}")

# # Define folder path
# folder_path = r"/home/anik/Research/Jujube Leaf/Coding Files/boroi_pata"
# count_non_square_images(folder_path)
