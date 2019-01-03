import cv2

image = cv2.imread("example.jpg")
print(image.shape)
cv2.imshow("示例图片", image)
cv2.waitKey(0)