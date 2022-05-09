import cv2
import matplotlib.pyplot as plt
from ImageDepth import ImageDepth

image_depth = ImageDepth()

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(image_depth.depth_from_image(img))
    plt.show()