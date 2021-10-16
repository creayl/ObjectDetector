import cv2 as cv
import numpy as np
import tensorflow as tf

import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WindowCapture(None)

# load the trained model
#cascade_tree = cv.CascadeClassifier("../cascade_classifier/cascade/cascade.xml")
# load an empty Vision class
vision_tree = Vision(None)

cv.namedWindow(
    "output", flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
)

# Read the graph.
with tf.compat.v1.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.compat.v1.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    loop_time = time()
    while True:

        # get an updated image of the game
        screenshot = wincap.get_screenshot()

        # Read and preprocess an image.
        rows = screenshot.shape[0]
        cols = screenshot.shape[1]
        inp = cv.resize(screenshot, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(screenshot, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

        # do object detection
        #rectangles = cascade_tree.detectMultiScale(screenshot)

        # draw the detection results onto the original image
        #detection_image = vision_tree.draw_rectangles(screenshot, rectangles)

        # display the images
        cv.imshow("output", screenshot)

        # debug the loop rate
        print("FPS {}".format(1 / (time() - loop_time)))
        loop_time = time()

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        key = cv.waitKey(1)
        if key == ord("q"):
            cv.destroyAllWindows()
            break

    print("Done.")