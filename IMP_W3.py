#### mask
import cv2
import numpy as np
from ultralytics.utils import imwrite

img = cv2.imread("pic/Lena.png")

masksize = (25,25) ##left 25 right 25 up 25 down 25
anchor = (500,500)  ## up right 150 +25

## มิติแรก แนวตั้ง 2 แนวนอน
xstart = anchor[0]- (int)(masksize[0]/2)
ystart = anchor[1]- (int)(masksize[1]/2)
xstop =  xstart   + masksize[0]
ystop =  ystart   + masksize[1]

mark_point = img[ystart:ystop,xstart:xstop ,:].copy()
mark_area = img.copy()
mark_area[ystart:ystop,xstart:xstop ,:]=[0,255,0]
print(xstop)
print(ystop)

# cv2.imshow("ori", img)
# cv2.imshow("area", mark_area)
# cv2.imshow("Output", mark_point)

#save image
#cv2.imwrite("xxx500.png", mark_point)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



############################ section  2
kernalavg = np.array([
[1/9, 1/9, 1/9],
[1/9, 1/9, 1/9],
[1/9, 1/9, 1/9] ], dtype=np.float32)


# kernalcon = kernalavg[::-1,::-1]

## low pass filter (blur)
filtered_avg = cv2.filter2D(img,-1,kernalavg,borderType= cv2.BORDER_REFLECT) ## zero padding
cv2.imshow("Output_filtered", filtered_avg)
##cv2.imwrite("xfilteravg0padding.png", filtered_avg)
cv2.imshow("orignal", img)
cv2.waitKey(0)
cv2.destroyAllWindows()






###3
blurred  = cv2.GaussianBlur(img,(5,5),0)
shappened =cv2.addWeighted(img,1.8,blurred,-0.8,0)  ## must sum is 1 (1.5,-0.5)

cv2.imshow("ori", img)
cv2.imshow("shap", shappened)
cv2.waitKey(0)
cv2.destroyAllWindows()