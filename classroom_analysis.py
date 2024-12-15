# -*- coding: utf-8 -*-
"""classroom_analysis.py"""

# Path to the image you are using for analysis
img_name = 'C:/Users/rakes/Downloads/unnam44ed.jpg'

# Import necessary libraries
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os

# Define the model path relative to the script location
model_path = os.path.join(os.path.dirname(__file__), "model_files")

# Check if CUDA (GPU) is available, otherwise fallback to CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Specify the path to your locally downloaded model or use a model from Hugging Face
# Make sure the path contains the required model files (config.json, pytorch_model.bin, etc.)
# Change this to your local model directory

# Load the model and processor from the local path
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Define your prompt
prompt = "<CBMS>"

# Load the image using PIL (Ensure the path is correct)
image = Image.open(img_name).convert("RGB")

# Process the inputs (text and image)
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

# Generate text based on the model
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)

# Decode the generated text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# Post-process the generated text (task-specific post-processing)
parsed_answer = processor.post_process_generation(generated_text, task="<CBMS>", image_size=(image.width, image.height))

# Print the parsed answer
print(parsed_answer)

# Display the image using PIL
image.show()
input("Press Enter to exit...")
