import os
import re

def rename_files_sequentially(folder_path, start_name):
    # Extract prefix and start number from the input
    match = re.match(r'(\D*)(\d+)', start_name)
    if not match:
        print("Error: The start_name must include a numeric suffix (e.g., 'img1').")
        return
    
    prefix, num = match.groups()
    current_number = int(num)

    files = sorted(os.listdir(folder_path))

    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(file)
            new_name = f"{prefix}{current_number}{file_extension}"
            new_file_path = os.path.join(folder_path, new_name)
            
            # Avoid overwriting existing files
            if not os.path.exists(new_file_path):
                os.rename(file_path, new_file_path)
                current_number += 1
            else:
                print(f"Skipping {file}, target file {new_name} already exists.")

# Example usage:
folder = input("Enter the folder path: ")
start_name = input("Enter the starting filename (e.g., img1): ")
rename_files_sequentially(folder, start_name)
