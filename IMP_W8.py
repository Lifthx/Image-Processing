import cv2
from ultralytics.utils import imwrite
import numpy as np

img = cv2.imread(r"C:\Users\llift\cropped_hands\img_hand0.jpg",cv2.IMREAD_GRAYSCALE)
T_low = 110
T_high = 170
mask =cv2.inRange(img,T_low,T_high)
output=img *(mask/255)
edges = cv2.Canny(mask, 110, 170)

cv2.imshow("original",img)
cv2.imshow("mask",mask)
cv2.imshow("output",output.astype(np.uint8))
cv2.imshow("edges",edges)
imwrite("output110170.jpg",mask)
imwrite("edges.jpg",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

##section 2
# img = cv2.imread("pic/sudoku.png",cv2.IMREAD_GRAYSCALE)
#
# retval,thresh = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
#
# thresh_mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,0)
#
# thresh_gaussian = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,0)
#
# cv2.imshow("single",thresh)
# cv2.imshow("mean",thresh_mean)
# cv2.imshow("gaussian",thresh_gaussian)
# imwrite("thresh.jpg",thresh)
# imwrite("thresh_mean.jpg",thresh_mean)
# imwrite("thresh_gaussian.jpg",thresh_gaussian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##section 3
# img = cv2.imread("pic/road.jpg", cv2.IMREAD_GRAYSCALE)
# blurred  =cv2.GaussianBlur(img,(7,7),1.5)
#
# low_t=130
# high_t=200
#
#
#
# edges = cv2.Canny(blurred,low_t,high_t)
#
# cv2.imshow("Original", img)
# cv2.imshow("Blurred", blurred)
# cv2.imshow("Edges", edges)
# imwrite("canny.jpg", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
