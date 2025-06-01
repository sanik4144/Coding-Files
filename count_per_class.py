import os

def count_images_in_folders(folder_path):
    # Dictionary to store the count of images in each subfolder
    image_count_per_folder = {}

    for subdir, dirs, files in os.walk(folder_path):
        # Skip the root folder itself (if it's not a subfolder)
        if subdir == folder_path:
            continue

        image_count = 0
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_count += 1

        # Print the count of images in the current subfolder (class)
        if image_count > 0:
            subfolder_name = os.path.basename(subdir)
            image_count_per_folder[subfolder_name] = image_count
            print(f"{subfolder_name}: {image_count} images")

    # Optionally, print total image count
    total_images = sum(image_count_per_folder.values())
    print(f"\nTotal images in all folders: {total_images}")

# Define the folder path
folder_path = r"/home/anik/Research/Jujube Leaf/Coding Files/boroi_pata"
count_images_in_folders(folder_path)
