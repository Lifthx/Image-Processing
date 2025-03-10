import cv2
import numpy as np

# แสดงค่า Kernel ของ Sobel และ Scharr
kx_sobel, ky_sobel = cv2.getDerivKernels(1, 0, 3, normalize=False)
sobel_x_kernel = np.outer(kx_sobel, ky_sobel)

kx_sobel, ky_sobel = cv2.getDerivKernels(0, 1, 3, normalize=False)
sobel_y_kernel = np.outer(kx_sobel, ky_sobel)

kx_scharr, ky_scharr = cv2.getDerivKernels(1, 0, cv2.FILTER_SCHARR, normalize=False)
scharr_x_kernel = np.outer(kx_scharr, ky_scharr)

kx_scharr, ky_scharr = cv2.getDerivKernels(0, 1, cv2.FILTER_SCHARR, normalize=False)
scharr_y_kernel = np.outer(kx_scharr, ky_scharr)

print("Sobel X Kernel:\n", sobel_x_kernel)
print("\nSobel Y Kernel:\n", sobel_y_kernel)
print("\nScharr X Kernel:\n", scharr_x_kernel)
print("\nScharr Y Kernel:\n", scharr_y_kernel)
