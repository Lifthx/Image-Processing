import cv2
import numpy as np
original = cv2.imread("pic/Lena.png").astype(np.int16)

result_3 = original*3
result_2 = original*2

result_3=np.clip(result_3,0,255)
result_2=np.clip(result_2,0,255)
print(original.dtype)

cv2.imshow("original",original.astype(np.uint8))
cv2.imshow("result*2",result_2.astype(np.uint8))
cv2.imshow("result*3",result_3.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
print(original)




# Invert the original image
inverted = 255 - original.astype(np.uint8)
inverted = np.clip(inverted, 0, 255)
# Print the data type of the original image
print(original.dtype)

# Display the images
cv2.imshow("Original", original.astype(np.uint8))
cv2.imshow("Result * 3", result_3.astype(np.uint8))
cv2.imshow("Inverted", inverted.astype(np.uint8))

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt
import numpy as np


ct = cv2.imread("pic/CT_sample 1.tif")
ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)


cv2.imshow("CT", ct)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(ct)

sted = cv2.normalize(ct, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
hist = cv2.calcHist([ct], [0], None, [256], [0, 256])
print(hist)


y = hist.ravel()
x = np.arange(256)
plt.bar(x, y)
plt.title("Histogram")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


cv2.imshow("Normalized CT", sted)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt
import numpy as np

ct = cv2.imread("pic/CT_sample 1.tif")
ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)

cv2.imshow("CT", ct)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(ct)

sted = cv2.normalize(ct, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

hist = cv2.calcHist([ct], [0], None, [256], [0, 256])
normalized_hist = cv2.calcHist([sted], [0], None, [256], [0, 256])
print("hist")
print(hist)
print("Normalized Histogram")
print(normalized_hist)

plt.figure(figsize=(12, 6))

# Plot original histogram
plt.subplot(1, 2, 1)
plt.bar(np.arange(256), hist.ravel(), color='blue', alpha=0.7)
plt.title("Original Histogram")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.bar(np.arange(256), normalized_hist.ravel(), color='green', alpha=0.7)
plt.title("Normalized Histogram")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

cv2.imshow("Normalized CT", sted)
cv2.waitKey(0)
cv2.destroyAllWindows()

eqaulized = cv2.equalizeHist(ct)

hist2 = cv2.calcHist([eqaulized], [0], None, [256], [0, 256])
print(hist2)


y = hist2.ravel()
x = np.arange(256)
plt.bar(x, y)
plt.title("equalized Histogram")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


cv2.imshow("equlized ", sted)
cv2.waitKey(0)
cv2.destroyAllWindows()