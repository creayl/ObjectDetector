import cv2
import numpy as np
import os

from windowcapture import WindowCapture

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their
# own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Threshold to detect object
thres = 0.5
nms_threshold = 0.2

useWebcam = False
wincap = WindowCapture(None)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("avail. classes: ", classNames)
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "ssd_mobilenet_v3_large_frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    if useWebcam:
        success, img = cap.read()

    else:
        img = wincap.get_screenshot()

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    # print(type(confs[0]))
    # print(confs)

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    # print(indices)

    for i in indices:
        # i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, h + y),
                      color=(0, 255, 0), thickness=2)
        cv2.putText(
            img,
            classNames[classIds[i] - 1].upper(),
            (box[0] + 10, box[1] + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            str(round(confs[i] * 100, 2)),
            (box[0] + 300, box[1] + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("output", img)
    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break

print("Done.")
