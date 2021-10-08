import cv2 as cv
import numpy as np
import os
import win32gui
from time import time
from windowcapture import WindowCapture
from detection import Detection
from vision import Vision
from bot import NewWorldBot, BotState

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their
# own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEBUG = True


def winEnumHandler(hwnd, ctx):
    if win32gui.IsWindowVisible(hwnd):
        print(hex(hwnd), win32gui.GetWindowText(hwnd))


if DEBUG:
    win32gui.EnumWindows(winEnumHandler, None)


# initialize the WindowCapture class
wincap = WindowCapture(None)
# load the detector
# TODO Find out how to build our own CascadeClassifier model for opencv, then we could use this Detection class, you can probably find this under https://www.youtube.com/watch?v=KecMlLUuiE4&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI&index=1&ab_channel=LearnCodeByGaming
# detector = Detection('harvestable_model_final.xml')
# load an empty Vision class
# vision = Vision()
# initialize the bot
# bot = NewWorldBot((wincap.offset_x, wincap.offset_y), (wincap.w, wincap.h))

wincap.start()
# detector.start()
# bot.start()

# Threshold to detect object
thres = 0.5
nms_threshold = 0.2

classNames = []
classFile = "../nets/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("avail. classes: ", classNames)
configPath = "../nets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "../nets/ssd_mobilenet_v3_large_frozen_inference_graph.pb"

net = cv.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cv.namedWindow(
    "output", flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
)

while True:
    # if we don't have a screenshot yet, don't run the code below this point yet
    if wincap.screenshot is None:
        continue

    # give detector the current screenshot to search for objects in
    # detector.update(wincap.screenshot)

    # update the bot with the data it needs right now
    # if bot.state == BotState.INITIALIZING:
    #     # while bot is waiting to start, go ahead and start giving it some targets to work
    #     # on right away when it does start
    #     targets = vision.get_click_points(detector.rectangles)
    #     bot.update_targets(targets)
    # elif bot.state == BotState.SEARCHING:
    #     # when searching for something to click on next, the bot needs to know what the click
    #     # points are for the current detection results. it also needs an updated screenshot
    #     # to verify the hover tooltip once it has moved the mouse to that position
    #     targets = vision.get_click_points(detector.rectangles)
    #     bot.update_targets(targets)
    #     bot.update_screenshot(wincap.screenshot)
    # elif bot.state == BotState.MOVING:
    #     # when moving, we need fresh screenshots to determine when we've stopped moving
    #     bot.update_screenshot(wincap.screenshot)
    # elif bot.state == BotState.MINING:
    #     # nothing is needed while we wait for the mining to finish
    #     pass

    # if DEBUG:
    #     # draw the detection results onto the original image
    #     detection_image = vision.draw_rectangles(wincap.screenshot, detector.rectangles)
    #     # display the images
    #     cv.imshow('Matches', wincap.screenshot)

    img = wincap.screenshot
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        cv.putText(
            img,
            classNames[classIds[i][0] - 1].upper(),
            (box[0] + 10, box[1] + 30),
            cv.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv.putText(
            img,
            str(round(confs[i] * 100, 2)),
            (box[0] + 200, box[1] + 30),
            cv.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv.imshow("output", img)

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord("q"):
        wincap.stop()
        cv.destroyAllWindows()
        break

print("Done.")
