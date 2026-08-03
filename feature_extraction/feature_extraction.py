# Batch process all images and extract embeddings
import pandas as pd
from tqdm import tqdm
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModel
import torch
import requests
import numpy as np
import os 
import glob
import argparse
import nibabel as nib


# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract DINO v2 embeddings from medical images')
parser.add_argument('--img_dir', type=str, 
                    default='//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_processed/Dataset999_AutoPet/nnUNetPlans_3d_fullres/',
                    help='Directory containing image files')
parser.add_argument('--file_type', type=str, choices=['npy', 'nii.gz'], default='npy',
                    help='File type to process: "npy" or "nii.gz"')
parser.add_argument('--normalize', action='store_true',
                    help='Normalize the images before processing')
parser.add_argument('--output_name', type=str, default='embedding',
                    help='Name for the output files')

args = parser.parse_args()

IMG_DIR = args.img_dir
FILE_TYPE = args.file_type

# Validate directory
if not os.path.exists(IMG_DIR):
    raise ValueError(f"Image directory does not exist: {IMG_DIR}")

print(f"Image directory: {IMG_DIR}", flush=True)
print(f"File type: {FILE_TYPE}", flush=True)

# Create file pattern based on file type
if FILE_TYPE == 'npy':
    file_pattern = '*[0-9].npy'
elif FILE_TYPE == 'nii.gz':
    file_pattern = '*_0001.nii.gz'
else:
    raise ValueError(f"Unknown file type: {FILE_TYPE}")

# Load image files
image_files = glob.glob(os.path.join(IMG_DIR, file_pattern))
print(f"Found {len(image_files)} {FILE_TYPE} files")

if len(image_files) == 0:
    raise ValueError(f"No {FILE_TYPE} files found in {IMG_DIR}")

print(f"Loading first image for testing...")
if FILE_TYPE == 'npy':
    image = np.load(image_files[0])
elif FILE_TYPE == 'nii.gz':
    image = nib.load(image_files[0]).get_fdata()
else:
    raise ValueError(f"Unknown file type: {FILE_TYPE}")

# # Load model and processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()

# Handle multi-dimensional medical image
print(f"Original image shape: {image.shape}")



print(f"Found {len(image_files)} images to process\n")

SPLIT_FILE = '//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/nnUNet_processed/Dataset999_AutoPet/splits_final.json'

import json 

with open(SPLIT_FILE, 'r') as f:
    split_data = json.load(f)

train_files = split_data[6]['train']
print(len(train_files))

embeddings_list = []
metadata_list = []

for img_path in tqdm(image_files):  # Process first 10 for testing
    if os.path.basename(img_path.replace('.npy', '').replace('_0001.nii.gz', '')) not in train_files:
        if os.path.exists(img_path):
            print("exists")
        else: 
            raise FileNotFoundError(f"File not found: {img_path}")
        try:
            # Load image based on file type
            if FILE_TYPE == 'npy':
                img_data = np.load(img_path)
            elif FILE_TYPE == 'nii.gz':
                import nibabel as nib
                img_data = nib.load(img_path).get_fdata()
            else:
                raise ValueError(f"Unknown file type: {FILE_TYPE}")
            
            # convert to 2D via a mean projection
            if img_data.ndim == 4:
                print(f"Image {img_path} has 4D shape {img_data.shape}, taking mean across modalities")
                
                img_2d = img_data[1].mean(axis=0)  # Mean across modalities
            elif img_data.ndim == 3:
                print(f"Image {img_path} has 3D shape {img_data.shape}, taking mean across depth")
                img_2d = img_data.mean(axis=0)  # Mean across depth
            else:
                img_2d = img_data
            
            # print the new shape and stats
            print(f"Processed image shape: {img_2d.shape}")
            
            # Normalize and convert to PIL RGB
            if args.normalize:
                img_2d_norm = ((img_2d - img_2d.min()) / (img_2d.max() - img_2d.min()) * 255).astype(np.uint8)
            else:
                img_2d_norm = img_2d.astype(np.uint8)
                
            pil_img = PILImage.fromarray(img_2d_norm, mode='L').convert('RGB')
            
            # Get embeddings
            inputs = processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract CLS token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # Calculate patch statistics
            patch_emb = outputs.last_hidden_state[:, 1:, :].cpu().numpy()
            patch_mean = patch_emb.mean(axis=1).flatten()
            patch_std = patch_emb.std(axis=1).flatten()
            
            embeddings_list.append({
                'filename': os.path.basename(img_path),
                'cls': cls_embedding,
                'patch_mean': patch_mean,
                'patch_std': patch_std
            })
            
            metadata_list.append({
                'filename': os.path.basename(img_path),
                'shape': img_data.shape,
                'img_min': float(img_2d.min()),
                'img_max': float(img_2d.max()),
                'cls_mean': float(cls_embedding.mean()),
                'cls_std': float(cls_embedding.std())
            })
            
        except Exception as e:  
            print(f"Error processing {img_path}: {e}")
    
    else: 
        print(f"Skipping {img_path}")

# Create metadata DataFrame
metadata_df = pd.DataFrame(metadata_list)
print(f"\nProcessed {len(metadata_df)} images successfully")
print(f"\nMetadata summary:")
print(metadata_df.head())
print(f"\nEmbedding statistics across batch:")
print(f"CLS embedding - mean: {metadata_df['cls_mean'].mean():.4f}, std: {metadata_df['cls_std'].mean():.4f}")
# save the metadata_df

normalize_stat = '_normalized_' if args.normalize else '_'
format_stat = '_npy_' if FILE_TYPE == 'npy' else '_niigz_'


metadata_df.to_csv(args.output_name + '_metadata.csv', index=False)

embeddings_df = pd.DataFrame(embeddings_list)
print(f"\nEmbeddings DataFrame:")
print(embeddings_df.head())
embeddings_df.to_pickle(args.output_name + '_embeddings.pkl')  # Save embeddings for later use


print("\nFeature extraction completed successfully!")
