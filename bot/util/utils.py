import tensorflow as tf
from time import time
from typing import List
from numpy.testing._private.nosetester import NoseTester

from domain.boundingbox import BoundingBox
from util.vision import Vision


class Utils:

    isBushVision: Vision = None
    isFlintVision: Vision = None
    isFreshWaterVision: Vision = None

    # constructor
    def __init__(self):
        self.isBushVision = Vision("img/bush.png")
        self.isFlintVision = Vision("img/flint.png")
        self.isFreshWaterVision = Vision("img/fresh_water.png")

    def loadModel(self):
        print("Loading model...", end="")
        start_time = time()

        # Load saved model and build the detection function
        detect_fn = tf.saved_model.load("saved_model")

        end_time = time()
        elapsed_time = end_time - start_time
        print("Done! Took {} seconds".format(elapsed_time))
        return detect_fn

    def calculateBestBox(self, bboxes: List[BoundingBox], screenSize):
        if len(bboxes) == 0:
            return None

        bestBox = bboxes[0]
        for box in bboxes:
            if bestBox.score(screenSize) < box.score(screenSize):
                bestBox = box
        return bestBox

    def calculateNearestBox(self, bboxes: List[BoundingBox], screenWidth):
        if len(bboxes) == 0:
            return None

        bestBox = bboxes[0]
        for box in bboxes:
            bestBoxDist = abs(self.calculateMoveDistance(bestBox, screenWidth))
            boxDist = abs(self.calculateMoveDistance(box, screenWidth))
            if bestBoxDist > boxDist:
                bestBox = box
        return bestBox

    def calculateMoveDistance(self, bestBox: BoundingBox, screenWidth):
        if bestBox == None:
            return 0
        # is on right side of screen
        boxMiddle = bestBox.pointStart.x + bestBox.width / 2
        screenMiddle = screenWidth / 2
        moveDistance = boxMiddle - screenMiddle
        return moveDistance

    def cropImage(self, img, x, y, width, height):
        return img[y : y + height, x : x + width]

    def isBush(self, screenshot: NoseTester):
        return len(self.isBushVision.find(screenshot, threshold=0.95)) > 0

    def isFlint(self, screenshot: NoseTester):
        return len(self.isFlintVision.find(screenshot, threshold=0.95)) > 0
    
    def isFreshWater(self, screenshot: NoseTester):
        return len(self.isFreshWaterVision.find(screenshot, threshold=0.95)) > 0
