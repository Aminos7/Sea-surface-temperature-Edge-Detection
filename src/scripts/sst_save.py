import os
import numpy as np
import matplotlib.pyplot as plt

# Load data
sst_path = 'C:/Users/AmineSnoussi/Desktop/SST/SST_train.npy'

sst_train = np.load(sst_path)

# Create output directory if it doesn't exist
output_dir = 'train_sst'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save images
for i in range(sst_train.shape[0]):
    sst_img_path = os.path.join(output_dir, f'sst_{i}.png')
    
    sst_img = sst_train[i,:,:]
    
    # Normalize image values
    sst_min, sst_max = np.nanmin(sst_img), np.nanmax(sst_img)
    sst_img = (sst_img - sst_min) / (sst_max - sst_min)
    
    # Convert to 8-bit integers
    sst_img = (sst_img * 255).astype(np.uint8)
    
    # Save image
    plt.imsave(sst_img_path, sst_img, cmap='jet')
    
    print(f'Saved image {sst_img_path}.')
