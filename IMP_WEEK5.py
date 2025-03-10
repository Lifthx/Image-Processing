import cv2
import numpy
import numpy as np
import noise_generator_for_lab05

image = cv2.imread(r"C:\Users\llift\Downloads\honey.jpg", cv2.IMREAD_GRAYSCALE)
#
# image2 = noise_generator_for_lab05.salt_and_pepper_noise(image,0.10,0.10)
#
# cv2.imshow("image after de noise ",image2)
# #cv2.imwrite("cameraman 0.15.jpg",image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ## gaussian

# image3 = noise_generator_for_lab05.gaussian_noise(image,0,15)
#
# cv2.imshow("gaussian image ",image3)
# #cv2.imwrite("cameraman  gaussian 225...jpg",image3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#sparkle
# image_sparkle = noise_generator_for_lab05.sparkle_noise(image,0,50)
#
# cv2.imshow("spakle image",image_sparkle)
# #cv2.imwrite("cameraman  sparkle 50p.jpg",image4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# ## denoise median
# denoisemedian = cv2.medianBlur(image2,5)
# cv2.imshow("denoisemedian",denoisemedian)
# cv2.imwrite("denoisemedian_5.jpg",denoisemedian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# ## denoise avg
# denoiseavg = cv2.blur(image2,(1000,1000))#it not work it make noise "smear"
# cv2.imshow("denoiseavg",denoiseavg)
# cv2.imwrite("denoiseavg_1000.jpg",denoiseavg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#avg noise
#import most of gaussian noise from last week
# imgmostgau = cv2.imread(r"C:\Users\llift\Desktop\image processing\cameraman  gaussian 50 with calmean.jpg")
# cv2.imshow("imgmostgau",imgmostgau)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_gau_list = []
for i in range(1000):
    image_gau_list.append(noise_generator_for_lab05.gaussian_noise(image,0,50))


guassianremove = np.mean(image_gau_list,axis=0).astype(np.uint8)#axis =0 is for avg of every point
print(guassianremove)
cv2.imshow("removednoise from gauss",guassianremove)
#cv2.imwrite("removednoise from gauss1000naja.jpg",guassianremove)
cv2.waitKey(0)
cv2.destroyAllWindows()

# #log
# img_log = np.log1p(np.float32(image_sparkle))
# #must normalrise when you show image
# ss=cv2.normalize(img_log,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
# cv2.imshow("sparklelog",cv2.normalize(img_log,None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
# cv2.imwrite("sparkle log.jpg",ss)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# #speckle log time 0 shirt is back no have noise
# out_speckle = np.expm1(img_log).astype(np.uint8)
# cv2.imshow("out_speckle",out_speckle)
# cv2.imwrite("out_speckle.jpg",out_speckle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()