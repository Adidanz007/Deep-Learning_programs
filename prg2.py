#2. Write a Python program to rotate and translate a given image using affine transformation. Display the original image, rotated image, translated image, and combined rotated and translated image.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols = img.shape[:2]

# -------- Rotation --------
angle = 45
center = (cols//2, rows//2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))

# -------- Translation --------
tx, ty = 60, 70
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, translation_matrix, (cols, rows))

# -------- Rotation + Translation --------
combined_matrix = rotation_matrix.copy()
combined_matrix[0,2] += tx
combined_matrix[1,2] += ty

combined = cv2.warpAffine(img, combined_matrix, (cols, rows))

# -------- Display Images --------
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(rotated)
plt.title("Rotated Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(translated)
plt.title("Translated Image")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(combined)
plt.title("Rotated + Translated")
plt.axis("off")

plt.show()