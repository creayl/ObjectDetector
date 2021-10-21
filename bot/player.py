import pyautogui
import cv2 as cv
import numpy as np
import random
import win32api, win32con
from numpy.testing._private.nosetester import NoseTester
from time import sleep, time

from util.windowcapture import WindowCapture
from util.vision import Vision
from util.logger import Logger
from util.utils import Utils


class Player:

    # constants
    # escape
    KEY_EXIT_BOT = 27
    # to be defined
    # KEY_PAUSE_BOT = ''

    KEY_HARVEST = "e"
    KEY_WALK_BACKWARDS = "s"
    KEY_AUTO_WALK = ","

    KEY_FIRST_ATTACK = "q"
    KEY_SECOND_ATTACK = "r"
    KEY_THIRD_ATTACK = "f"

    KEY_FIRST_POTION = "3"
    KEY_SECOND_POTION = "4"
    KEY_THIRD_POTION = "5"
    KEY_FOURTH_POTION = "6"

    # properties
    pyautogui.FAILSAFE = False
    is_autowalk = False
    should_end = False

    canHarvestVision: Vision = None
    stillHarvestingVision: Vision = None

    wincap: WindowCapture = None
    logger: Logger = None
    utils: Utils = None

    # constructor
    def __init__(self):
        self.wincap = WindowCapture(None)
        self.canHarvestVision = Vision("img/harvest_tooltip.png")
        self.stillHarvestingVision = Vision("img/harvest_in_progress.png")
        self.utils = Utils()
        self.logger = Logger()

    def setMoving(self, move: bool):
        if self.is_autowalk != move:
            pyautogui.press(self.KEY_AUTO_WALK)
            self.is_autowalk = not self.is_autowalk

    def takeALook(self):
        return self.wincap.get_screenshot()

    def fightResponse(self):
        pyautogui.press(self.KEY_FIRST_ATTACK)
        if self.shouldEnd(0.5):
            return
        pyautogui.press(self.KEY_FIRST_POTION)
        if self.shouldEnd(0.5):
            return

    def buffYourself(self):
        pyautogui.press(self.KEY_SECOND_POTION)
        if self.shouldEnd(0.5):
            return
        pyautogui.press(self.KEY_THIRD_POTION)
        if self.shouldEnd(0.5):
            return
        pyautogui.press(self.KEY_FOURTH_POTION)
        if self.shouldEnd(0.5):
            return

    def isInFight(self, screenshot: NoseTester):
        img = self.utils.cropImage(screenshot, 1009, 1293, 250, 36)
        pixels = np.float32(img.reshape(-1, 3))

        n_colors = 5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        dominant = palette[np.argmax(counts)]
        # BGR
        return dominant[0] < 30 and dominant[1] < 30 and dominant[2] > 120

    def canHarvest(self, screenshot: NoseTester):
        return len(self.canHarvestVision.find(screenshot, threshold=0.8)) > 0

    def isHarvesting(self, screenshot: NoseTester):
        img = self.utils.cropImage(screenshot, 650, 350, 1150, 700)
        return len(self.stillHarvestingVision.find(img, threshold=0.6)) > 0

    def moveToTarget(self):
        # walk forward as long as screenshot is not detected
        self.setMoving(False)

        start_time = time()
        while True:
            screenshot = self.takeALook()

            can_harvest = self.canHarvest(screenshot)
            # is_harvesting = self.isHarvesting(screenshot)

            if can_harvest:
                self.setMoving(False)

            # if is_harvesting:
            #     self.setMoving(False)

            if self.isInFight(screenshot):
                self.fightResponse()

            if can_harvest:
                return True
            # if is_harvesting:
            #     return True

            if time() - start_time > 30:
                self.setMoving(False)
                self.moveMouseInFluidMotion(random.randrange(500, 1000))
                
                self.shouldEnd(0.5)
                return False

            self.logger.logToImage(
                screenshot, "MOVE: " + str(round(time() - start_time, 1))
            )

            cv.imshow("output", screenshot)

            self.setMoving(True)
            if self.shouldEnd():
                return False

    def waitForHarvest(self):
        self.setMoving(False)
        self.shouldEnd(0.5)

        parking_end = time() + random.uniform(2, 3)
        while time() < parking_end:
            screenshot = self.takeALook()

            if not self.canHarvest(screenshot):
                # einparken
                pyautogui.keyDown(self.KEY_WALK_BACKWARDS)
            else:
                pyautogui.keyUp(self.KEY_WALK_BACKWARDS)
                break

            # debug output
            self.logger.logToImage(
                screenshot, "EINPARKEN: " + str(round(parking_end - time(), 1))
            )
            cv.imshow("output", screenshot)

            if self.shouldEnd():
                pyautogui.keyUp(self.KEY_WALK_BACKWARDS)
                return

        pyautogui.keyUp(self.KEY_WALK_BACKWARDS)
        screenshot = self.takeALook()
        if not self.canHarvest(screenshot):
            return

        # harvest
        pyautogui.press(self.KEY_HARVEST)

        sleep_start = time()
        sleep_end = sleep_start + random.uniform(3, 5)

        while self.isHarvesting(self.takeALook()) or time() <= sleep_end:
            print("stillHarvesting...")
            screenshot = self.takeALook()

            if time() > sleep_start + 1.5 and self.canHarvest(screenshot):
                return

            # debug output
            self.logger.logToImage(
                screenshot, "HARVESTING: " + str(round(sleep_end - time(), 1))
            )
            cv.imshow("output", screenshot)

            if self.shouldEnd():
                return

            if self.isInFight(screenshot):
                self.fightResponse()

            if self.shouldEnd(0.25):
                return
            # sleep(0.25)

    def harvest(self, moveDistance):
        if self.shouldEnd():
            return
        self.moveMouseInFluidMotion(moveDistance)
        self.shouldEnd(1)

        if not self.moveToTarget():
            return

        while True:
            self.waitForHarvest()
            screenshot = self.takeALook()
            if not self.canHarvest(screenshot):
                break

    def moveMouseInFluidMotion(self, moveDistance):
        distanceMoved = 0
        movePerTick = 10
        while distanceMoved < moveDistance:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(movePerTick), 0, 0, 0)
            distanceMoved = distanceMoved + movePerTick
            sleep(0.001)

    def shouldEnd(self, delay=0.001):
        # convert delay from ms to sec
        delay = delay * 1000
        if self.should_end:
            return True
        key = cv.waitKey(int(delay))
        # escape pressed
        if key == self.KEY_EXIT_BOT:
            self.setMoving(False)
            cv.destroyAllWindows()
            self.should_end = True
            return True
