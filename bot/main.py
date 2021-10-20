import os
from typing import List
import cv2 as cv
import tensorflow as tf
import random
from time import sleep, time
from util.windowcapture import WindowCapture
from util.vision import Vision
from domain.boundingbox import BoundingBox
import win32api, win32con
from player import Player
from util.utils import Utils
from util.logger import Logger

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize classes
wincap = WindowCapture(None)
vision = Vision("img/harvest_tooltip.png")
player = Player(wincap, vision)
utils = Utils()
logger = Logger()

cv.namedWindow(
    "output", flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
)

detect_fn = utils.loadModel()

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

    bestBox = utils.calculateBestBox(bboxList, int(cols * rows))
    moveDistance = utils.calculateMoveDistance(bestBox, cols)

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
        logger.logToImageWithColorAndCoordinates(
            screenshot,
            str(round(confidence * 100, 2)) + "%",
            color,
            (int(x + 4), int(y + 50)),
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
        player.harvest(moveDistance)
    else:  # fail-safe
        win32api.mouse_event(
            win32con.MOUSEEVENTF_MOVE, random.randrange(500, 1000), 0, 0, 0
        )
        sleep(0.5)

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if player.shouldEnd():
        break
print("Done.")
