# STAGE 4: Feature Extraction
from transformers import ViTImageProcessor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Normalizes laser reflections to 224x224 pixels
