import pyautogui
import cv2 as cv
from time import sleep, time
from numpy.testing._private.nosetester import NoseTester
import numpy as np
from util.windowcapture import WindowCapture
from util.vision import Vision
import random
import win32api, win32con


class Player:

    # properties
    pyautogui.FAILSAFE = False
    is_autowalk = False
    should_end = False

    vision: Vision = None
    wincap: WindowCapture = None

    # constructor
    def __init__(self, wincap, vision):
        self.wincap = wincap
        self.vision = vision

    def setMoving(self, move: bool):
        if self.is_autowalk != move:
            pyautogui.press(",")
            self.is_autowalk = not self.is_autowalk

    def fightResponse(self):
        pyautogui.press("q")
        sleep(0.5)
        if self.shouldEnd():
            return
        pyautogui.press("3")
        sleep(0.5)
        if self.shouldEnd():
            return

    def isInFight(self, screenshot: NoseTester):
        x = 1009
        y = 1293
        w = 250  # 532
        h = 36
        img = screenshot[y : y + h, x : x + w]
        pixels = np.float32(img.reshape(-1, 3))

        n_colors = 5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        dominant = palette[np.argmax(counts)]
        # BGR
        return dominant[0] < 30 and dominant[1] < 30 and dominant[2] > 120

    def hasTarget(self, screenshot):
        return len(self.vision.find(screenshot, threshold=0.8)) > 0

    def moveToTarget(self):
        # walk forward as long as screenshot is not detected
        self.setMoving(False)

        start_time = time()
        while True:
            screenshot = self.wincap.get_screenshot()

            has_target = self.hasTarget(screenshot)
            if has_target:
                self.setMoving(False)

            if self.isInFight(screenshot):
                self.fightResponse()

            if has_target:
                return True

            if time() - start_time > 30:
                self.setMoving(False)
                win32api.mouse_event(
                    win32con.MOUSEEVENTF_MOVE, random.randrange(500, 1000), 0, 0, 0
                )
                sleep(0.5)
                self.shouldEnd()
                return False

            cv.putText(
                screenshot,
                "MOVE: " + str(round(time() - start_time, 1)),
                (int(20), int(50)),
                cv.FONT_HERSHEY_PLAIN,
                4,
                (50, 255, 50),
                2,
            )
            cv.imshow("output", screenshot)
            self.setMoving(True)
            if self.shouldEnd():
                return False

    def waitForHarvest(self):
        self.setMoving(False)
        sleep(0.5)

        parking_end = time() + random.uniform(2, 3)
        while time() < parking_end:
            screenshot = self.wincap.get_screenshot()

            if not self.hasTarget(screenshot):
                # einparken
                pyautogui.keyDown("s")
            else:
                pyautogui.keyUp("s")
                break

            # debug output
            cv.putText(
                screenshot,
                "EINPARKEN: " + str(round(parking_end - time(), 1)),
                (int(20), int(190)),
                cv.FONT_HERSHEY_PLAIN,
                4,
                (50, 255, 50),
                2,
            )
            cv.imshow("output", screenshot)

            if self.shouldEnd():
                pyautogui.keyUp("s")
                return

        pyautogui.keyUp("s")
        screenshot = self.wincap.get_screenshot()
        if not self.hasTarget(screenshot):
            return
        pyautogui.press("e")

        sleep_start = time()
        sleep_end = sleep_start + random.uniform(6, 8)
        while time() <= sleep_end:
            screenshot = self.wincap.get_screenshot()

            if time() > sleep_start + 1.5 and self.hasTarget(screenshot):
                return

            # debug output
            cv.putText(
                screenshot,
                "HARVESTING: " + str(round(sleep_end - time(), 1)),
                (int(20), int(120)),
                cv.FONT_HERSHEY_PLAIN,
                4,
                (50, 255, 50),
                2,
            )
            cv.imshow("output", screenshot)

            if self.shouldEnd():
                return

            if self.isInFight(screenshot):
                self.fightResponse()

            if self.shouldEnd():
                return
            sleep(0.25)

    def harvest(self, moveDistance):
        if self.shouldEnd():
            return
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(moveDistance), 0, 0, 0)
        sleep(1)

        if not self.moveToTarget():
            return

        while True:
            self.waitForHarvest()
            screenshot = self.wincap.get_screenshot()
            if not self.hasTarget(screenshot):
                break

    def shouldEnd(self):
        if self.should_end:
            return True
        key = cv.waitKey(1)
        # escape pressed
        if key == 27:
            self.setMoving(False)
            cv.destroyAllWindows()
            self.should_end = True
            return True