import cv2
import matplotlib.pyplot as plt
import pandas

from ImageDepth import ImageDepth
from PersonDetection import PersonDetection

image_depth = ImageDepth()
person_detector = PersonDetection()

cap = cv2.VideoCapture(0)

frame = 0
while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    person = person_detector.detect(img)
    depth = image_depth.depth_from_image(img)
    # Normalise between 0-1
    depth /= (depth.max()/1.0)

    cv2.rectangle(depth, (int(person.xmin), int(person.ymin)), (int(person.xmax), int(person.ymax)), (255,0,0), 10)

    #depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGRA)
    cv2.imshow('image', depth)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()