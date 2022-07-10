import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json
import sys, os

from sympy import O



class CrossStitch:
    def __init__(self):
        with open("color_ref.json", 'r') as file:
            self.colorRef = json.load(file)


    def load_image(self, fileName, bgColor=(0, 255, 0)):
        img = cv2.imread(fileName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([1, 1, 1]))
        img[mask > 0] = bgColor

        self.img = img
        self.imgDim = img.reshape((img.shape[0] * img.shape[1], 3))

        # plt.figure()
        # plt.axis("off")
        # plt.imshow(img)
        # plt.show()


    def centroid_histogram(self, nClusters):
        clt = KMeans(n_clusters=nClusters)
        clt.fit(self.imgDim)
        # based on the number of pixels assigned to each cluster
        # grab the number of different clusters and create a histogram
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        self.clt = clt
        self.hist = hist


    def prepare_colors(self):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        # loop over the percentage of each cluster and the color of
        # each cluster

        for (percent, color) in zip(self.hist, self.clt.cluster_centers_):
            if (np.round(color) == np.array([0, 255, 0])).all():
                pFact = percent

        colors = []

        for (percent, color) in zip(self.hist, self.clt.cluster_centers_):
            if (np.round(color) != np.array([0, 255, 0])).all():
                # plot the relative percentage of each cluster
                # endX = startX + (percent * 300)
                endX = startX + (percent * (1 + pFact / (1 - pFact)) * 300)
                cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                            color.astype("uint8").tolist(), -1)
                startX = endX

                colors.append(color)

        self.bar = bar
        self.colors = colors


    def prepare_threads(self):
        idxRef = []
        idxDict = "0"
        for c in self.colors:
            minVal = 255

            for idxC in self.colorRef:
                if np.mean(np.abs(c - self.colorRef[idxC]["RGB"])) <= minVal:
                    minVal = np.mean(np.abs(c - self.colorRef[idxC]["RGB"]))
                    idxDict = idxC

            idxRef.append(idxDict)

        # imgsAnchor, imgsDmc = [], []

        # fig, axs = plt.subplots(2, len(idxRef))
        for i, item in enumerate(idxRef):
            imgsAnchor.append(mpimg.imread(f"color_ref/Anchor/anchor{self.colorRef[item]['Anchor']}.jpg"))
            imgsDmc.append(mpimg.imread(f"color_ref/DMC/dmc{self.colorRef[item]['DMC']}.jpg"))


            # axs[0, i].imshow(imgsAnchor[i])
            # axs[1, i].imshow(imgsDmc[i])

        plt.show()



    def make_plots(self):
        pass




# anchorFiles = os.listdir("color_ref/Anchor")
# dmcFiles = os.listdir("color_ref/DMC")

# for a in anchorFiles:
#     anchor = a.replace("Anchor-", "anchor")
#     # os.path.join("color_ref/Anchor", a)
#     os.rename(os.path.join("color_ref/Anchor", a), os.path.join("color_ref/Anchor", anchor))

# for d in dmcFiles:
#     dmc = d.replace("117mc_e_", "dmc")
#     dmc = dmc.replace("_swatch_150x150", "")
#     os.rename(os.path.join("color_ref/DMC", d), os.path.join("color_ref/DMC", dmc))




if __name__=="__main__":
    cs = CrossStitch()
    cs.load_image("c:\\users\\sena\\downloads\\jujuba_bg.png")
    cs.centroid_histogram(7)
    cs.prepare_colors()
    cs.prepare_threads()