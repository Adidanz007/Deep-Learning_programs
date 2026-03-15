#3. Write a Python program to apply various image smoothing techniques on a noisy image using Gaussian Blur and Median Filter and compare the effects.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# -------- Add Noise --------
noise = np.random.normal(0, 25, gray.shape)
noisy_img = gray + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# -------- Gaussian Blur --------
gaussian = cv2.GaussianBlur(noisy_img, (5,5), 0)

# -------- Median Filter --------
median = cv2.medianBlur(noisy_img, 5)

# -------- Display Images --------
plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(noisy_img, cmap='gray')
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(gaussian, cmap='gray')
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(median, cmap='gray')
plt.title("Median Filter")
plt.axis("off")

plt.show()