import cv2
import numpy as np


def ideal_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff:
                mask[i, j] = 1.0
    return mask

def ideal_high_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance > cutoff:
                mask[i, j] = 1.0
    return mask



def butterworth_low_pass_filter(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 / (1 + (distance / cutoff) ** (2 * order))
    return mask

def butterworth_high_pass_filter(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if(distance==0):
                mask[i, j] = 0
            else:
                mask[i, j] = 1 / (1 + (cutoff/distance) ** (2 * order))
    return mask

def apply_filter(img,filter_mask):
    img_float = np.float32(img)
    dft = cv2.dft(img_float,flags=cv2.DFT_COMPLEX_OUTPUT)
    print(dft.shape)
    dft_shift=np.fft.fftshift(dft)
    mask_2ch =cv2.merge([filter_mask,filter_mask]) #cause dft have 2 channel but filter mask have 1
    fshift = dft_shift * mask_2ch
    fshift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(fshift)
    img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    cv2.normalize(img_back,img_back,0,255,cv2.NORM_MINMAX) #cause value more than limit
    img_back = np.uint8(img_back)

    return img_back
