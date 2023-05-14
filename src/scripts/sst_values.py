import numpy as np

# Load the npy file
sst_train = np.load('C:/Users/ASUS/Desktop/SST/SST_train.npy')

# Print the minimum and maximum sst values
print(f'Minimum sst value: {np.nanmin(sst_train)}')
print(f'Maximum sst value: {np.nanmax(sst_train)}')
