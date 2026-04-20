# STAGE 4: Feature Extraction(Load raw-ds, processor, transforms)

#Step 6: Load Datasets and Check Column Names

from datasets import load_dataset, Image as HFImage

raw_ds = load_dataset(
    "imagefolder",
    data_dir=MASTER_DIR,
    drop_labels=False
)

raw_ds = raw_ds.cast_column("image", HFImage(decode=True))
print(raw_ds)

# Remap HuggingFace alphabetical indices to correct numerical indices
# hf_to_correct was built in Step 4B
def remap_label(example):
    example['label'] = hf_to_correct[example['label']]
    return example

raw_ds = raw_ds.map(remap_label)
print("Labels remapped to correct numerical indices")
print(raw_ds)

#Step 7: Defining Processor

from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
print("Processor loaded!")

# Step 8: Transform

# Original whole-image transform for baseline and Phase 1

from PIL import Image as PILImage
from transformers import ViTImageProcessor

# Load processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
print("Processor loaded!")

def transform(example_batch):
    inputs = processor(
        [x.convert("RGB") for x in example_batch['image']],
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

prepared_ds = raw_ds.with_transform(transform)
print("Dataset transformed and ready!")
