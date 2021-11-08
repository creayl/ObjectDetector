import tensorflow as tf
from time import time
from typing import List
from numpy.testing._private.nosetester import NoseTester
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

    def cropImage(self, img, x, y, width, height):
        return img[y : y + height, x : x + width]

    def isBush(self, screenshot: NoseTester):
        return len(self.isBushVision.find(screenshot, threshold=0.95)) > 0

    def isFlint(self, screenshot: NoseTester):
        return len(self.isFlintVision.find(screenshot, threshold=0.95)) > 0

    def isFreshWater(self, screenshot: NoseTester):
        return len(self.isFreshWaterVision.find(screenshot, threshold=0.95)) > 0
