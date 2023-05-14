import numpy as np
import matplotlib.pyplot as plt

def load_data():
    sst_path  = 'C:/Users/AmineSnoussi/Desktop/SST/SST_train.npy'
    mask_path = 'C:/Users/AmineSnoussi/Desktop/SST/MASK_train.npy'

    sst_train  = np.load(sst_path)
    mask_train = np.load(mask_path, allow_pickle=True)

    mask_train = np.nan_to_num(mask_train, nan=1)

    return sst_train, mask_train

sst_train, mask_train = load_data()

index = np.random.randint(sst_train.shape[0])

plt.figure(figsize=(15, 10))

plt.subplot(121)
plt.imshow(sst_train[index,:,:], cmap='jet')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.title('SST')

plt.subplot(122)
plt.imshow(mask_train[index,:,:], cmap='binary')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.title('Artificial Ground Truth')

plt.show()
