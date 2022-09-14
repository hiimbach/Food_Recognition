import cv2
from infer import FoodDetection
import os


model = FoodDetection()
for img in os.listdir("sample_image"):
    if img.split(".")[-1] == "jpg":
        sample_image = cv2.imread("sample_image/" + img)
        result = model.detect(sample_image)
        print(result)