import torch

class PersonDetection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def detect(self, img):
        results = self.model(img)

        # Results
        df = results.pandas().xyxy[0]
        df = df.loc[df["name"] == "person"]
        return df.loc[df["confidence"].idxmax()]