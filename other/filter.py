import os
from PIL import Image
from collections import Counter
import shutil

def get_png_resolutions(directory):
    resolutions = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                    resolutions.append((width, height))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return resolutions

def delete_non_16x16_or_error_pngs(directory):
    deleted_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                    if (width, height) != (16, 16):
                        os.remove(file_path)
                        deleted_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"Deleted {file_path} due to error.")
                    except Exception as remove_error:
                        print(f"Could not delete {file_path}: {remove_error}")
    return deleted_files

def count_16x16_pngs(directories):
    count = 0
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.png'):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                        if (width, height) == (16, 16):
                            count += 1
                    except Exception:
                        continue
    return count

def flatten_and_rename_pngs(parent_folder):
    for category in ["blocks", "items"]:
        category_path = os.path.join(parent_folder, category)
        if not os.path.isdir(category_path):
            continue
        for mod_folder in os.listdir(category_path):
            mod_folder_path = os.path.join(category_path, mod_folder)
            if not os.path.isdir(mod_folder_path):
                continue
            for root, _, files in os.walk(mod_folder_path):
                for file in files:
                    if file.lower().endswith('.png'):
                        rel_path = os.path.relpath(root, mod_folder_path)
                        # Get the image name (without any subfolders)
                        image_name = file
                        # If there are subfolders, add them to the name
                        if rel_path != ".":
                            subfolder_part = rel_path.replace(os.sep, "_")
                            new_name = f"{mod_folder}_{subfolder_part}_{image_name}"
                        else:
                            new_name = f"{mod_folder}_{image_name}"
                        src = os.path.join(root, file)
                        dst = os.path.join(category_path, new_name)
                        # Avoid overwriting files with the same name
                        if os.path.exists(dst):
                            base, ext = os.path.splitext(new_name)
                            i = 1
                            while os.path.exists(os.path.join(category_path, f"{base}_{i}{ext}")):
                                i += 1
                            dst = os.path.join(category_path, f"{base}_{i}{ext}")
                        shutil.move(src, dst)
            # Optionally, remove the now-empty mod folder
            for root, dirs, _ in os.walk(mod_folder_path, topdown=False):
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
            if not os.listdir(mod_folder_path):
                os.rmdir(mod_folder_path)

if __name__ == "__main__":
    assets_dir = "assets"
    resolutions = get_png_resolutions(assets_dir)
    resolution_counts = Counter(resolutions)
    for res, count in resolution_counts.most_common():
        print(f"{res[0]}x{res[1]}: {count} files")
    deleted = delete_non_16x16_or_error_pngs(assets_dir)
    print(f"Deleted {len(deleted)} files not 16x16 or with errors.")
    dirs_to_check = ["items", "blocks"]
    count = count_16x16_pngs(dirs_to_check)
    print(f"Total 16x16 PNG files in 'items' and 'blocks' folders and subfolders: {count}")
    flatten_and_rename_pngs("assets")
    print("Flattened and renamed all PNGs in assets/blocks and assets/items.")

