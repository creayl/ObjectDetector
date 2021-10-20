from time import sleep, time
import tensorflow as tf
from domain.boundingbox import BoundingBox
from typing import List
import cv2 as cv


class Utils:

    def loadModel(self):
        print("Loading model...", end="")
        start_time = time()

        # Load saved model and build the detection function
        detect_fn = tf.saved_model.load('saved_model')

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

    def calculateMoveDistance(self, bestBox: BoundingBox, screenWidth):
        if bestBox == None:
            return 0
        # is on right side of screen
        boxMiddle = bestBox.pointStart.x + bestBox.width / 2
        screenMiddle = screenWidth / 2
        moveDistance = boxMiddle - screenMiddle
        return moveDistance
