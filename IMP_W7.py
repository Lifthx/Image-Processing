import cv2
import numpy as np
from ultralytics.utils import imwrite

img = cv2.imread("pic/pout2.tif",cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
##magnitude


magnitude = cv2.magnitude(dft[:,:,0],dft[:,:,1])
#logscale
magnitude_log = np.log(1+magnitude)
#normalize
freq = cv2.normalize(magnitude_log,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

dft_shift = np.fft.fftshift(dft)  # if you must dft to use next must be shift all of dft  dft 2 chanel # real # chengson
#mask
magnitude_shift = np.fft.fftshift(freq) # shift freq cuase we just want to show

row,column = img.shape

centerrow = row//2
centercolumn = column//2

mask = np.ones((row, column), np.uint8)


r=37
cv2.circle(mask,(centercolumn,centerrow),r,0,4) ## r is raduis #0 line is black 5 is tthickness // dont swap center column row


dft_shift[:,:,0] = dft_shift[:,:,0] * mask #real value
dft_shift[:,:,1] = dft_shift[:,:,1] * mask
magnitude_shift = magnitude_shift/255.0 * mask  #for only show


dft_in =np.fft.ifftshift(dft_shift)
idft = cv2.idft(dft_in,flags=cv2.DFT_REAL_OUTPUT)
result = cv2.normalize(idft,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow("original",img)
cv2.imshow("shifted ",magnitude_shift)
cv2.imshow("filtered",result)
imwrite("manitude.png",magnitude_shift.astype(np.uint8))
imwrite("filtered.png",result)
cv2.waitKey(0)
cv2.destroyAllWindows()

