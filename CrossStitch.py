import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import json


class CrossStitch:
    def __init__(self):
        with open("color_ref.json", 'r') as file:
            colorRef = json.load(file)


    def load_image(self, fileName, bgColor=(0, 255, 0)):
        img = cv2.imread(fileName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([1, 1, 1]))
        img[mask > 0] = bgColor

        plt.figure()
        plt.axis("off")
        plt.imshow(img)
        plt.show()


if __name__=="__main__":
    CrossStitch().load_image("c:\\users\\sena\\downloads\\jujuba_bg.png")