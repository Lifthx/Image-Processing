import cv2
import numpy as np
from ultralytics.utils import imwrite

img = cv2.imread(r"C:\Users\llift\AppData\Roaming\StabilityMatrix\Packages\stable-diffusion-webui-forge\outputs\txt2img-images\2024-12-14\00007-3587504207.png",cv2.IMREAD_GRAYSCALE)

##section 1
#dft furier is just this one
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)



##magnitude
magnitude = cv2.magnitude(dft[:,:,0],dft[:,:,1])
print("magnitude = ",magnitude[0,0])
#logscale
magnitude_log = np.log(1+magnitude)

#normalize
freq = cv2.normalize(magnitude_log,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

##
shifted_magnitude = np.fft.fftshift(freq)



cv2.imshow("original", img)
cv2.imshow("shifted_magnitude", shifted_magnitude)
cv2.imwrite("shifted_magnitude_trumpfeedtrhump.png",shifted_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

##section3
invertdft = cv2.idft(dft,flags=cv2.DFT_REAL_OUTPUT)
resultinvertdft = cv2.normalize(invertdft,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow("resultinvertdft",resultinvertdft)
imwrite("resultinvertdft_oftrumpfeedtrhump.png",resultinvertdft)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite("spatialim.png", img)
##we get real and imaginary values but if want to show must have 3 channel is value of color those two is not
