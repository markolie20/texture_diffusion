import ollama
import base64

MODEL_NAME = "gemma3:4b" # The model that's failing
# Replace with a path to one of your 16x16 PNGs
IMAGE_PATH = "assets/blocks/actuallyadditions_block_tiny_torch.png"

def image_to_base64_list(image_path):
    with open(image_path, "rb") as img_file:
        return [base64.b64encode(img_file.read()).decode('utf-8')]

print(f"Testing model {MODEL_NAME} with image {IMAGE_PATH}")
try:
    b64_images = image_to_base64_list(IMAGE_PATH)
    response = ollama.generate(
        model=MODEL_NAME,
        prompt="This is a minecraft texture, what do you think it depicts.",
        images=b64_images,
        stream=False,
        options={"temperature": 0.1, "num_predict": 50},
    )
    print("LLM Response:")
    print(response.get('response', 'No response content found.'))
    print("\nFull response dictionary:")
    print(response)

except ollama.ResponseError as e:
    print(f"Ollama API Response Error: {e.status_code} - {e.error}")
    print(f"Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")