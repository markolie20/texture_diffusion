import os
import zipfile
import shutil
from pathlib import Path

def extract_mod_textures(mods_folder_path, output_folder_path):
    """
    Extracts block and item textures from .jar Minecraft mods.

    Args:
        mods_folder_path (str or Path): Path to the folder containing .jar mod files.
        output_folder_path (str or Path): Path to the folder where textures will be extracted.
    """
    mods_folder = Path(mods_folder_path)
    output_folder = Path(output_folder_path) # This will be 'assets'

    if not mods_folder.is_dir():
        print(f"Error: Mods folder '{mods_folder}' not found.")
        print("Please ensure the path is correct and the folder exists.")
        return

    output_blocks_folder = output_folder / "blocks"
    output_items_folder = output_folder / "items"
    output_blocks_folder.mkdir(parents=True, exist_ok=True)
    output_items_folder.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for .jar files in: {mods_folder}")
    jar_files_found = 0
    textures_extracted_count = 0

    for jar_file_path in mods_folder.glob("*.jar"):
        jar_files_found += 1
        mod_name = jar_file_path.stem
        print(f"\nProcessing mod: {mod_name}.jar")
        mod_textures_found_this_jar = 0

        try:
            with zipfile.ZipFile(jar_file_path, 'r') as jar_archive:
                for member_info in jar_archive.infolist():
                    if member_info.is_dir():
                        continue

                    entry_path_str_original = member_info.filename
                    # Normalize path separators and work with lowercase for matching
                    entry_path_str_normalized = entry_path_str_original.replace('\\', '/')
                    
                    if not entry_path_str_normalized.lower().endswith(".png"):
                        continue

                    # Split the path into components for easier searching
                    # e.g., "assets/modid/textures/block/stone.png" -> ["assets", "modid", "textures", "block", "stone.png"]
                    path_components = Path(entry_path_str_normalized).parts
                    path_components_lower = [part.lower() for part in path_components]
                    
                    try:
                        # Find the 'assets' directory
                        assets_index = path_components_lower.index("assets")
                    except ValueError:
                        # 'assets' not in path, skip this file
                        continue

                    try:
                        # Find the 'textures' directory *after* 'assets'
                        # Search in the slice of the list starting after 'assets'
                        textures_index_relative_to_assets = path_components_lower[assets_index + 1:].index("textures")
                        textures_index_absolute = assets_index + 1 + textures_index_relative_to_assets
                    except ValueError:
                        # 'textures' not found after 'assets', skip this file
                        continue

                    # The path parts relevant for categorization and output structure
                    # are those *after* the 'textures' folder.
                    # e.g., if path is 'assets/modid/textures/block/stone.png',
                    # sub_path_parts will be ('block', 'stone.png')
                    sub_path_parts = path_components[textures_index_absolute + 1:]
                    
                    if not sub_path_parts: # Should not happen if it's a file inside 'textures'
                        continue
                    
                    relative_texture_path = Path(*sub_path_parts) # e.g., Path('block/stone.png') or Path('foo/item/myitem.png')

                    # Determine category based on directory names within this relative_texture_path
                    # We look at parent directories of the texture file, within the 'textures' folder context
                    category_check_dirs = [part.lower() for part in relative_texture_path.parts[:-1]] # All parts except the filename

                    category = None
                    if "block" in category_check_dirs or "blocks" in category_check_dirs:
                        category = "blocks"
                    elif "item" in category_check_dirs or "items" in category_check_dirs:
                        category = "items"

                    if category:
                        # Output structure: output_folder/category/mod_name/relative_texture_path_structure
                        category_base_folder = output_folder / category
                        final_output_dir = category_base_folder / mod_name / relative_texture_path.parent
                        final_output_path = final_output_dir / relative_texture_path.name

                        final_output_dir.mkdir(parents=True, exist_ok=True)

                        with jar_archive.open(member_info) as source, open(final_output_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                            textures_extracted_count += 1
                            mod_textures_found_this_jar += 1
            
            if mod_textures_found_this_jar > 0:
                print(f"  Extracted {mod_textures_found_this_jar} textures from this mod.")

        except zipfile.BadZipFile:
            print(f"  Warning: Could not open {jar_file_path.name} as a zip file. It might be corrupted or not a valid JAR.")
        except Exception as e:
            print(f"  Error processing {jar_file_path.name}: {e}")
            # import traceback # Uncomment for debugging specific errors
            # traceback.print_exc()

    if jar_files_found == 0:
        print(f"No .jar files found in '{mods_folder}'.")
    else:
        print(f"\nExtraction complete. Found {jar_files_found} JAR files.")
        print(f"Extracted a total of {textures_extracted_count} block/item textures.")
        print(f"Textures saved in: {output_folder.resolve()}")
        print(f"  Block textures: {output_blocks_folder.resolve()}")
        print(f"  Item textures:  {output_items_folder.resolve()}")

if __name__ == "__main__":
    output_dir_name = 'assets'
    mods_folder_location = r'C:\Users\Mark\curseforge\minecraft\Instances\All the Mods 10 - ATM10\mods'
    
    current_script_directory = Path(__file__).resolve().parent
    output_directory = current_script_directory / output_dir_name

    print(f"Output will be saved to: {output_directory.resolve()}")
    extract_mod_textures(mods_folder_location, output_directory)