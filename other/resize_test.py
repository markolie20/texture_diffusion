import os
from PIL import Image

input_folder = 'to_resize'
output_folder = 'resized'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img_resized = img.resize((16, 16), Image.NEAREST)
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)