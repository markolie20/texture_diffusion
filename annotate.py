from google import genai
from dotenv import load_dotenv
from PIL import Image
from collections import Counter
from constants import TYPES, SUBTYPES, MATERIALS, COLOUR_NAMES, COLOUR_RGB_MAP
import os, math
from pydantic_classes import BaseModel
from enum import Enum



load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key= GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.0-flash"
ASSETS_FOLDER = "assets"
OUTPUT_FOLDER = "annotations"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "blocks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "items"), exist_ok=True)


def get_closest_palette_color(rgb_tuple, palette_rgb_map):
    min_dist = float('inf')
    if not palette_rgb_map: return closest_color_name

    r1, g1, b1 = rgb_tuple
    for name, palette_rgb in palette_rgb_map.items():
        r2, g2, b2 = palette_rgb
        dist = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
        if dist < min_dist:
            min_dist = dist
            closest_color_name = name
    return closest_color_name

def get_prominent_color(image_path, palette_rgb_map):
    img = Image.open(image_path).convert("RGBA")
    pixels = list(img.getdata())
    palette_counts = Counter()
    valid_pixels = 0
    for r, g, b, a in pixels:
        if a > 50: # Consider only non-transparent or semi-transparent pixels
            closest_palette_name = get_closest_palette_color((r, g, b), palette_rgb_map)
            palette_counts[closest_palette_name] += 1
            valid_pixels +=1
    if not palette_counts or valid_pixels == 0:
        return "transparent" # If all pixels are transparent or image is empty
    return palette_counts.most_common(1)[0][0]

