import os
from typing import List
import cv2 as cv
import numpy as np
from numpy.testing._private.nosetester import NoseTester
import tensorflow as tf
import pyautogui
import random
from time import sleep, time
from vision import Vision
from windowcapture import WindowCapture
from domain.boundingbox import BoundingBox
import win32api, win32con
from pynput import keyboard

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WindowCapture(None)

vision = Vision("img/harvest_tooltip.png")

cv.namedWindow(
    "output", flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
)

print("Loading model...", end="")
start_time = time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load("saved_model")

end_time = time()
elapsed_time = end_time - start_time
print("Done! Took {} seconds".format(elapsed_time))

pyautogui.FAILSAFE = False
should_end = False
is_autowalk = False

def shouldEnd():
    global should_end
    if should_end:
        return True
    key = cv.waitKey(1)
    if key == 27:
        setMoving(False)
        # pyautogui.keyUp("w")
        cv.destroyAllWindows()
        should_end = True
        return True


def calculateBestBox(bboxes: List[BoundingBox], screenSize):
    if len(bboxes) == 0:
        return None

    bestBox = bboxes[0]
    for box in bboxes:
        if bestBox.score(screenSize) < box.score(screenSize):
            bestBox = box
    return bestBox


def calculateMoveDistance(bestBox: BoundingBox, screenWidth):
    if bestBox == None:
        return 0
    # is on right side of screen
    boxMiddle = bestBox.pointStart.x + bestBox.width / 2
    screenMiddle = screenWidth / 2
    moveDistance = boxMiddle - screenMiddle
    return moveDistance


def fightResponse():
    pyautogui.press("q")
    sleep(0.5)
    if shouldEnd():
        return
    pyautogui.press("3")
    sleep(0.5)
    if shouldEnd():
        return
    pyautogui.press("5")
    sleep(0.5)
    if shouldEnd():
        return
    pyautogui.press("6")
    sleep(0.5)
    if shouldEnd():
        return


def isInFight(screenshot: NoseTester):
    x = 1009
    y = 1293
    w = 250  # 532
    h = 36
    img = screenshot[y : y + h, x : x + w]
    # average = img.mean(axis=0).mean(axis=0)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv.KMEANS_RANDOM_CENTERS

    # kmeans(pixels, n_colors, criteria, 10, flags)
    _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    # BGR
    return dominant[0] < 30 and dominant[1] < 30 and dominant[2] > 120


def hasTarget(screenshot):
    return len(vision.find(screenshot, threshold=0.8)) > 0


def setMoving(move: bool):
    global is_autowalk
    if is_autowalk != move:
        pyautogui.press(",")
        is_autowalk = not is_autowalk


def moveToTarget():
    # press w as long as screenshot is not detected
    setMoving(False)

    start_time = time()
    while True:
        screenshot = wincap.get_screenshot()

        has_target = hasTarget(screenshot)
        if has_target:
            setMoving(False)
            # pyautogui.keyUp("w")

        if isInFight(screenshot):
            fightResponse()

        if has_target:
            return True

        if time() - start_time > 30:
            setMoving(False)
            # pyautogui.keyUp("w")
            win32api.mouse_event(
                win32con.MOUSEEVENTF_MOVE, random.randrange(500, 1000), 0, 0, 0
            )
            sleep(0.5)
            shouldEnd()
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
        # pyautogui.press("w")
        # pyautogui.keyDown("w")
        setMoving(True)
        # sleep(random.uniform(3.5, 5.5))
        # pyautogui.keyUp("w")
        # sleep(0.1)
        if shouldEnd():
            return False


def waitForHarvest():
    setMoving(False)
    # pyautogui.keyUp("w")
    sleep(0.5)

    parking_end = time() + random.uniform(2, 3)
    while time() < parking_end:
        screenshot = wincap.get_screenshot()

        if not hasTarget(screenshot):
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

        if shouldEnd():
            pyautogui.keyUp("s")
            return

    pyautogui.keyUp("s")
    screenshot = wincap.get_screenshot()
    if not hasTarget(screenshot):
        return
    pyautogui.press("e")

    sleep_start = time()
    sleep_end = sleep_start + random.uniform(6, 8)
    while time() <= sleep_end:
        screenshot = wincap.get_screenshot()

        if time() > sleep_start + 1.5 and hasTarget(screenshot):
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

        if shouldEnd():
            return

        if isInFight(screenshot):
            fightResponse()

        if shouldEnd():
            return
        sleep(0.25)


def harvest(moveDistance):
    if shouldEnd():
        return
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(moveDistance), 0, 0, 0)
    sleep(1)

    if not moveToTarget():
        return

    while True:
        waitForHarvest()
        screenshot = wincap.get_screenshot()
        if not hasTarget(screenshot):
            break


loop_time = time()
while True:
    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    # Read and preprocess an image.
    rows = screenshot.shape[0]
    cols = screenshot.shape[1]
    inp = cv.resize(screenshot, (320, 320))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(inp)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections
    bboxList: List[BoundingBox] = []

    # Visualize detected bounding boxes.
    for i in range(num_detections):
        # classId = int(out[3][0][i])
        confidence = detections["detection_scores"][i]

        bbox = [float(v) for v in detections["detection_boxes"][i]]
        if confidence > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows

            # add bbox to list of boxes
            bboxList.append(BoundingBox(x, y, right, bottom, confidence))

    bestBox = calculateBestBox(bboxList, int(cols * rows))
    moveDistance = calculateMoveDistance(bestBox, cols)

    # Visualize detected bounding boxes.
    for i in range(len(bboxList)):
        bbox = bboxList[i]
        x = bbox.pointStart.x
        y = bbox.pointStart.y
        right = bbox.pointEnd.x
        bottom = bbox.pointEnd.y
        confidence = bbox.confidence
        color = (255, 100, 50)

        if confidence > 0.5:
            color = (125, 255, 51)
        if bbox == bestBox:
            color = (50, 50, 255)

        cv.rectangle(
            screenshot,
            (int(x), int(y)),
            (int(right), int(bottom)),
            color,
            thickness=2,
        )
        cv.putText(
            screenshot,
            str(round(confidence * 100, 2)) + "%",
            (int(x + 4), int(y + 50)),
            cv.FONT_HERSHEY_PLAIN,
            2,
            color,
            2,
        )
    cv.rectangle(
        screenshot,
        (int(cols / 2), int(rows / 2 - 1)),
        (int(cols / 2 + moveDistance), int(rows / 2 + 1)),
        (50, 50, 255),
        thickness=2,
    )
    # display the images
    cv.imshow("output", screenshot)
    # debug the loop rate
    print("FPS {}".format(1 / (time() - loop_time)))
    loop_time = time()

    if bestBox != None:
        harvest(moveDistance)
    else:  # fail-safe
        win32api.mouse_event(
            win32con.MOUSEEVENTF_MOVE, random.randrange(500, 1000), 0, 0, 0
        )
        sleep(0.5)

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if shouldEnd():
        break
print("Done.")
