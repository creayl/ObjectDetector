import os
import cv2 as cv
import numpy
import random
from time import time

from tensorflow.python.util.tf_inspect import signature

from player import Player
from util.utils import Utils
from util.logger import Logger
from util.ai import AI

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

player = Player()
utils = Utils()
logger = Logger()

DIST_MID_THRESH = 50

cv.namedWindow(
    "output", flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
)

ai = AI(utils.loadModel())


def analyseScreen():
    screenshot = player.takeALook()
    rows = screenshot.shape[0]
    cols = screenshot.shape[1]

    bboxList = ai.findTrees(screenshot)
    return screenshot, rows, cols, bboxList


def printDebugImage(screenshot, rows, cols, bboxList, bestBox, moveDistance, runtime):
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
    logger.logToImage(screenshot, "ROTATE: " + str(round(runtime, 1)))
    # display the images
    cv.imshow("output", screenshot)


loop_time = time()
buff_time = loop_time + random.randrange(5, 10)
while True:
    if time() > buff_time:
        player.buffYourself()
        buff_time = time() + 19 * 60 + random.randrange(10, 300)

    box_time = time()
    moveDirection = None
    while True:
        # get an updated image of the game
        screenshot, rows, cols, bboxList = analyseScreen()

        if time() < box_time + 6:
            bestBox = utils.calculateBestBox(bboxList, int(cols * rows))
        elif time() < box_time + 12:
            bestBox = utils.calculateNearestBox(bboxList, cols)
        else:
            bestBox = None
            break

        moveDistance = utils.calculateMoveDistance(bestBox, cols)

        if moveDirection == None:
            moveDirection = numpy.sign(moveDistance)

        if numpy.sign(moveDistance) != moveDirection:
            moveDistance = moveDistance * -1

        # Visualize detected bounding boxes.
        printDebugImage(
            screenshot, rows, cols, bboxList, bestBox, moveDistance, time() - box_time
        )

        if abs(moveDistance) < DIST_MID_THRESH + 10:
            break

        pixelDistance = numpy.sign(moveDistance) * DIST_MID_THRESH
        player.moveMouseInFluidMotion(pixelDistance)

        if player.canHarvest(screenshot):
            player.harvest()
            break

        if player.isInFight(screenshot):
            player.fightResponse()
            break

        if player.shouldEnd(0.1):
            bestBox = None
            break

    # debug the loop rate
    print("FPS {}".format(1 / (time() - loop_time)))
    loop_time = time()

    if bestBox != None:
        player.harvest()
    else:  # fail-safe
        player.moveMouseInFluidMotion(random.randrange(500, 1000))
        player.shouldEnd(0.5)

    # press 'esc' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if player.shouldEnd():
        break
print("Done.")
