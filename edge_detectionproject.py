import cv2
from PIL import Image, ImageFilter

image = Image.open("pic/pout.tif").convert("L")  # แปลงเป็นขาวดำก่อน
blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
edges = blurred.filter(ImageFilter.FIND_EDGES)  # ตรวจจับขอบ
edges.show()  # แสดงภาพ


