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

#Step 7: Defining Processor
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
print("Processor loaded!")

#Step 8: Augmentation Transformation (Phase 2 fixed version)

# Step 8 REVISED — Fixed Augmentation
from PIL import Image as PILImage
from torchvision import transforms
import torch

# Augmentation ONLY — no ToTensor or Normalize here
train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
        ratio=(0.8, 1.2)
    ),
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 2.0)
    ),
    transforms.RandomAdjustSharpness(
        sharpness_factor=2,
        p=0.3
    ),
    # NO ToTensor or Normalize — processor handles this
])

def transform_train(example_batch):
    augmented_images = [
        train_augmentation(img.convert("RGB")) 
        for img in example_batch['image']
    ]
    inputs = processor(
        images=augmented_images,
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

def transform_val(example_batch):
    inputs = processor(
        images=[img.convert("RGB") for img in example_batch['image']],
        return_tensors='pt'
    )
    inputs['labels'] = example_batch['label']
    return inputs

# Apply transforms
prepared_ds_train = raw_ds['train'].with_transform(transform_train)
prepared_ds_val   = raw_ds['validation'].with_transform(transform_val)
prepared_ds_test  = raw_ds['test'].with_transform(transform_val)

print("Augmentation pipeline ready!")
