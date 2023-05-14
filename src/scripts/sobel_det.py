import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image and convert it to grayscale
img = cv2.imread('C:/Users/AmineSnoussi/Desktop/SST/train_sst/sst_1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply the Sobel operator to compute the gradient in x and y directions
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Compute the magnitude of the gradient
mag = np.sqrt(sobelx**2 + sobely**2)

# Plot the original image and the magnitude of the gradient
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mag, cmap='gray')
plt.title('Magnitude of Gradient')
plt.axis('off')

plt.show()
