import cv2

# Test cv2.imshow
img = cv2.imread("test_image.jpg")
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()