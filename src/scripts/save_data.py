import os
import numpy as np
import matplotlib.pyplot as plt

# Load data
sst_path = 'sst_train.npy'
mask_path = 'mask_train.npy'

sst_train = np.load(sst_path)
mask_train = np.load(mask_path)

# Create output directory if it doesn't exist
output_dir = 'train'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save images
for i in range(sst_train.shape[0]):
    sst_img_path = os.path.join(output_dir, f'sst_{i}.png')
    mask_img_path = os.path.join(output_dir, f'mask_{i}.png')
    
    sst_img = sst_train[i,:,:]
    mask_img = mask_train[i,:,:]
    
    # Normalize image values
    sst_min, sst_max = sst_img.min(), sst_img.max()
    sst_img = (sst_img - sst_min) / (sst_max - sst_min)
    
    # Convert to 8-bit integers
    sst_img = (sst_img * 255).astype(np.uint8)
    mask_img = (mask_img * 255).astype(np.uint8)
    
    # Save images
    plt.imsave(sst_img_path, sst_img, cmap='jet')
    plt.imsave(mask_img_path, mask_img, cmap='gray')
    
    print(f'Saved images {sst_img_path} and {mask_img_path}.')
