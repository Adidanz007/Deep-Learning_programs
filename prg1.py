#1. Write a Python program to convert a given color image to grayscale and then to binary using thresholding. Display the original image, grayscale image, and binary image

import matplotlib.pyplot as plt
import numpy as np

# Read the color image
img = plt.imread("strawberry.jpg")

# Convert to grayscale
gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

# Convert grayscale to binary using threshold
threshold = 127
binary = np.where(gray > threshold, 255, 0)

# Display the images
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.show()