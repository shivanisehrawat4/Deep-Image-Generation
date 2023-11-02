# Install necessary packages
!pip install diffusers transformers accelerate xformers==0.0.16rc425

# Import Necessary Libraries
from diffusers import StableDiffusionPipeline
import torch

# Load the Pretrained Model and Configure the Pipeline
model_name = "dreamlike-art/dreamlike-photoreal-2.0"
image_generation_pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
image_generation_pipeline = image_generation_pipeline.to("cuda")

# Define Prompts and Prepare Image Storage
prompts = ["A serene mountain landscape at sunrise, misty, 8K resolution, tranquil, natural beauty",
           "A high-tech futuristic laboratory, 8K resolution, sci-fi, advanced technology, innovative",
           "Abstract representation of underwater worlds, vibrant colors, intricate patterns"]
generated_images = []

# Generate Images
for i, prompt_text in enumerate(prompts):
    generated_image = image_generation_pipeline(prompt_text).images[0]
    generated_image.save(f'generated_image_{i}.jpg')
    generated_images.append(generated_image)







