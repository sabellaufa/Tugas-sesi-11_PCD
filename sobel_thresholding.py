import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Membaca citra grayscale
# -------------------------------
image = imageio.imread("bella.JPG")
image = image.astype(np.float32)

# Konversi RGB ke grayscale jika perlu
if image.ndim == 3:
    image = image.mean(axis=2)

image = image.astype(np.float32)

# -------------------------------
# 2. Kernel Sobel
# -------------------------------
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# -------------------------------
# 3. Konvolusi manual
# -------------------------------
def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

gx = convolve(image, sobel_x)
gy = convolve(image, sobel_y)

# -------------------------------
# 4. Magnitude gradien
# -------------------------------
gradient_magnitude = np.sqrt(gx**2 + gy**2)

# Normalisasi
gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

# -------------------------------
# 5. Basic Thresholding
# -------------------------------
threshold = 50
segmented = np.zeros_like(gradient_magnitude)
segmented[gradient_magnitude >= threshold] = 255

# -------------------------------
# 6. Visualisasi
# -------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Citra Grayscale")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Deteksi Tepi Sobel")
plt.imshow(gradient_magnitude, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Hasil Thresholding")
plt.imshow(segmented, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
