import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, set_logging 
from utils.torch_utils import select_device


class FoodDetection:
    def __init__(self):
        # Initialize
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load("weights/final_model.pt", map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = 640
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

    @torch.inference_mode()
    def detect(self, original_image:np.ndarray):
        # Padded resize
        img = letterbox(original_image, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # class names
        names = ['Water', 'Pear', 'Egg', 'Butter', 'Apple', 'Cheese', 'Carrot', 'Onion', 'Banana', 'Mixed vegetables', 'Almonds', 'Dark chocolate', 'Biscuits', 'Corn', 'Pineapple', 'Peanut butter', 'Beer', 'Pumpkin', 'Mango', 'Cucumber', 'Pasta', 'Mayonnaise', 'Chicken', 'Orange', 'Ketchup', 'Milk', 'Tea', 'Black olives', 'Honey', 'Raspberries', 'Sour cream', 'Zucchini', 'Quinoa', 'Milk chocolate', 'Sausage', 'Lemon', 'Dates', 'Sesame seeds', 'Cottage cheese', 'Cream cheese', 'Blueberries', 'Cream', 'Bread', 'Strawberries' ]

        result = {}
        pred = pred[0]
        number_of_ingredient = pred.shape[0]
        for i in range(number_of_ingredient):
            ind = int(pred[i][5])
            result[f"{pred[i][4]}"] = names[ind]

        return result