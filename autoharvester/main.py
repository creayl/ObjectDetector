import os
import cv2 as cv
import numpy
import random
import pyautogui
from time import time
from numpy.testing._private.nosetester import NoseTester
from tensorflow.python.util.tf_inspect import signature
from util.utils import Utils
from util.vision import Vision
from util.windowcapture import WindowCapture

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap = WindowCapture(None)
utils = Utils()
canHarvestVision = Vision("img/harvest_tooltip.png")
stillHarvestingVision = Vision("img/harvest_in_progress.png")

# constants
# escape
KEY_EXIT_BOT = 27
# to be defined
# KEY_PAUSE_BOT = ''
KEY_HARVEST = "e"


def takeALook():
    return wincap.get_screenshot()


def canHarvest(screenshot: NoseTester):
    # img = self.utils.cropImage(screenshot, 650, 350, 1150, 700)
    if utils.isBush(screenshot):
        print("isBush")
        return False
    return len(canHarvestVision.find(screenshot, threshold=0.8)) > 0


def isHarvesting(screenshot: NoseTester):
    # img = self.utils.cropImage(screenshot, 650, 350, 1150, 700)
    return len(stillHarvestingVision.find(screenshot, threshold=0.6)) > 0


loop_time = time()
while True:
    while True:
        print("lookingForHarvestable")
        screenshot = takeALook()
        if not canHarvest(screenshot):
            print("cantHarvest")
            break

        # harvest
        print("canHarvest")
        pyautogui.press(KEY_HARVEST)

        sleep_start = time()
        sleep_min = sleep_start + random.uniform(2, 3)
        sleep_max = sleep_start + random.uniform(10, 15)

        while (
            isHarvesting(takeALook()) and not time() >= sleep_max
        ) or time() <= sleep_min:
            print("stillHarvesting...")
            screenshot = takeALook()

            if time() > sleep_start + 1.5 and canHarvest(screenshot):
                break
        print("harvestingDone")

    # debug the loop rate
    print("FPS {}".format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'esc' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == KEY_EXIT_BOT:
        break

print("Done.")
