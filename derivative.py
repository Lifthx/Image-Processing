import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color, exposure

# โหลดภาพ skimage
img_prewitt = io.imread(r"C:\Users\llift\Downloads\hands1.jpg")
img_prewitt = color.rgb2gray(img_prewitt)
# โหลดภาพด้วย
img_sobel_scharr = cv2.imread(r"C:\Users\llift\Downloads\hands1.jpg", cv2.IMREAD_GRAYSCALE)

# เบลอ scipy
img_prewitt = filters.gaussian(img_prewitt, sigma=1.4, mode='nearest')
# ใช้ cv2
img_sobel_scharr = cv2.GaussianBlur(img_sobel_scharr, (3, 3), 1.4)


# Prewitt (ใช้ skimage)
prewitt_combined = filters.prewitt(img_prewitt)
# Sobel (ใช้ OpenCV)
sobelx_3 = cv2.Sobel(img_sobel_scharr, cv2.CV_32F, 1, 0, ksize=3)
sobely_3 = cv2.Sobel(img_sobel_scharr, cv2.CV_32F, 0, 1, ksize=3)
sobel_combined_3 = np.sqrt(sobelx_3**2 + sobely_3**2)
# Scharr (ใช้ OpenCV)
scharrx = cv2.Scharr(img_sobel_scharr, cv2.CV_32F, 1, 0)
scharry = cv2.Scharr(img_sobel_scharr, cv2.CV_32F, 0, 1)
scharr_combined = np.sqrt(scharrx**2 + scharry**2)

# Canny
canny_edges = cv2.Canny(img_sobel_scharr, 70, 160)

# Normalization
prewitt_combined = exposure.rescale_intensity(prewitt_combined, in_range=(prewitt_combined.min(), prewitt_combined.max())
, out_range=(0, 255)).astype(np.uint8)
sobel_combined_3 = cv2.normalize(sobel_combined_3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
scharr_combined = cv2.normalize(scharr_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# เปรียบเทียบ Scharr - Sobel / Sobel - Prewitt
comparescharr_sobel = cv2.subtract(scharr_combined, sobel_combined_3)
comparesobel_prewitt = cv2.subtract(sobel_combined_3, prewitt_combined)


# บันทึกภาพ
cv2.imwrite("scharr-sobel.jpg", comparescharr_sobel)
cv2.imwrite("sobel-prewitt.jpg", comparesobel_prewitt)
cv2.imwrite("sobel.jpg", sobel_combined_3)
cv2.imwrite("scharr.jpg", scharr_combined)
cv2.imwrite("prewitt.jpg", prewitt_combined)

# แสดงผลเปรียบเทียบ
fig, axes = plt.subplots(3, 3, figsize=(12, 8))

axes[0, 0].imshow(img_sobel_scharr, cmap='gray')
axes[0, 0].set_title('Original Image (CV2)')

axes[0, 1].imshow(img_prewitt, cmap='gray')
axes[0, 1].set_title('Original Image (Skimage)')


axes[1, 0].imshow(prewitt_combined, cmap='gray')
axes[1, 0].set_title('Prewitt')

axes[1, 1].imshow(sobel_combined_3, cmap='gray')
axes[1, 1].set_title('Sobel')

axes[1, 2].imshow(scharr_combined, cmap='gray')
axes[1, 2].set_title('Scharr')

axes[2, 0].imshow(canny_edges, cmap='gray')
axes[2, 0].set_title('Canny')

axes[2, 1].imshow(comparesobel_prewitt, cmap='gray')
axes[2, 1].set_title('Sobel - Prewitt')

axes[2, 2].imshow(comparescharr_sobel, cmap='gray')
axes[2, 2].set_title('Scharr - Sobel')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
