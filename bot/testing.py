import pyautogui
import cv2 as cv
import numpy as np
import os
import random
import win32api, win32con
from numpy.testing._private.nosetester import NoseTester
from time import time

from util.windowcapture import WindowCapture
from util.vision import Vision
from util.logger import Logger
from util.utils import Utils

# This file is for testing code parts only.. dont remove useless imports in this file please. ;)
# i just wanna throw in code parts here and be able to run and test them fast


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# properties
pyautogui.FAILSAFE = False
is_autowalk = False
should_end = False
canHarvestVision: Vision = Vision("img/harvest_tooltip.png")
stillHarvestingVision: Vision = Vision("img/harvest_in_progress.png")
wincap: WindowCapture = WindowCapture(None)
logger: Logger = Logger()
utils: Utils = Utils()


def isStillHarvesting(screenshot: NoseTester):
    img = utils.cropImage(screenshot, 650, 350, 1150, 700)
    cv.imshow("debug", img)
    cv.waitKey(0)
    # return len(stillHarvestingVision.find(img, threshold=0.8)) > 0


isStillHarvesting(wincap.get_screenshot())
