# import cv2
# import numpy as np
# from ultralytics.utils import plt_settings
#
# img = cv2.imread("pic/Lena.png")
# cv2.imshow("Output", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img)
# print(img.shape)
#
# grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grayscale", grey)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# img = np.asarray(Image.open("pic/Lena.png"))
# plt.figure(num="openby plt")##ไม่จำเป็นไว้ตั้งชื่อรูปที่แสดง
# plt.axis('off') ##ไม่จำเป็น
# x = plt.imshow(img)#ต้องมีตัวแปรมาลองรับด้วย
# plt.show()

##แบบหลายหน้าต่างทีเดียว
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# อ่านภาพด้วย OpenCV
img = cv2.imread("pic/Lena.png")
# แปลงภาพเป็น Grayscale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# แปลงภาพ OpenCV (BGR) เป็น RGB เพื่อใช้กับ Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# สร้างหน้าต่างแสดงภาพทั้งหมด
plt.figure(figsize=(10, 5))

# แสดงภาพต้นฉบับในช่องย่อยที่ 1
plt.subplot(1, 3, 1)  # (rows, cols, index)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# แสดงภาพ Grayscale ในช่องย่อยที่ 2
plt.subplot(1, 3, 2)
plt.imshow(grey, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

# แสดงภาพที่เปิดด้วย PIL ในช่องย่อยที่ 3
img_pil = np.asarray(Image.open("pic/Lena.png"))
plt.subplot(1, 3, 3)
plt.imshow(img_pil)
plt.title("Image with PIL")
plt.axis('off')


# แสดงผลภาพทั้งหมดพร้อมกัน
plt.tight_layout()
plt.show()
