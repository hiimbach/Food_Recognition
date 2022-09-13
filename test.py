import cv2
from infer import FoodDetection


sample_image = cv2.imread("sample_image/vid_4_1840.jpg")
model = FoodDetection()

result = model.detect(sample_image)
print(result)