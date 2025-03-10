from ultralytics.utils import imwrite

from mask_filter_for_lab06 import *


original = cv2.imread("pic/Lena.png",cv2.IMREAD_GRAYSCALE)
# ##Ideal filtering
# cuttoff =100
# ideal_mask = ideal_high_pass_filter(original.shape,cuttoff)
#
# ideal_output =apply_filter(original,ideal_mask)
#
# cv2.imshow("original",original)
# cv2.imshow("ideal_mask",ideal_mask)
# cv2.imshow("ideal_output",ideal_output)
# imwrite("idealoutput100cutoff.jpg",ideal_output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##butterworth
cuttoff =100
order=2 ##canchange 1,2 or ....
butter_mask = butterworth_high_pass_filter(original.shape,cuttoff,order)

butter_output =apply_filter(original,butter_mask)

cv2.imshow("original",original)
cv2.imshow("butter_mask",butter_mask)
cv2.imshow("butter_output",butter_output)
imwrite("buuter100cutoff.jpg",butter_output)
cv2.waitKey(0)
cv2.destroyAllWindows()