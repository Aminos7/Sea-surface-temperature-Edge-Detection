import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio

import torch
import torchvision
import torch.nn as nn   
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from PIL import Image

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


#Collecting, assembling & resizing images for training

#Resize Dimensions:
dim_x = 208
dim_y = 108

#RINDNet output paths
rindnet_path = "C:/Users/AmineSnoussi/Downloads/Projet-Long-master/src/RINDNet pipeline/RINDNet-main"  #Main RINDNet directory
mat_path = rindnet_path + "/run/rindnet"                           #Matrices path
depth_path = mat_path + '/depth/mat'                               #Depth matrices directory
illu_path = mat_path + '/illumination/mat'                         #Illumination matrices directory
normal_path = mat_path + '/normal/mat'                             #Normal matrices directory
reflectance_path = mat_path + '/reflectance/mat'                   #Reflectance matrices directory

#path for the training/validation data
output_training_path = "C:/Users/AmineSnoussi/Downloads/Projet-Long-master/Training_Dataset/training_set"
output_validation_path = "C:/Users/AmineSnoussi/Downloads/Projet-Long-master/Training_Dataset/validation_set"


#number of images
n=len(os.listdir(depth_path))
i = 0

# Loop over the images in the depth directory:
for image in os.listdir(depth_path):
    try:
        # Reading the mat files:
        depth = sio.loadmat(os.path.join(depth_path, image)).get("result")
        illu = sio.loadmat(os.path.join(illu_path, image)).get("result")
        normal = sio.loadmat(os.path.join(normal_path, image)).get("result")
        reflec = sio.loadmat(os.path.join(reflectance_path, image)).get("result")

        # Assembling the 4 frames:
        result = np.concatenate((depth, illu, normal, reflec), axis=1)

        # Arrays transformations:
        img_array = (255 * (1 - result)).astype(np.uint8)
        img = Image.fromarray(img_array)

        # Image resizing:
        img = img.resize((dim_x, dim_y))

        # Splitting the dataset in 2 (training/validation):
        if i % 2 == 0:
            output_path = output_training_path
        else:
            output_path = output_validation_path

        # Saving the image:
        os.makedirs(output_path, exist_ok=True)
        img.save(os.path.join(output_path, image.replace(".mat", ".jpg")))

        # Printing the evolution of the process:
        i += 1
        if i % (n // 100) == 0:
            print(f"{i}/{n} ({100*i/n:.2f}%)")
    except Exception as e:
        print(f"Error processing file: {image}. Error message: {e}")

print("Transformation complete!")
