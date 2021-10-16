import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from time import time
from windowcapture import WindowCapture

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WindowCapture(None)

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

    # Visualize detected bounding boxes.
    for i in range(num_detections):
        # classId = int(out[3][0][i])
        score = detections["detection_scores"][i]
        # score = float(out[1][0][i])
        bbox = [float(v) for v in detections["detection_boxes"][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            color = (125, 255, 51)
            if score > 0.5:
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
                str(round(score * 100, 2)) + "%",
                (int(x + 4), int(y + 50)),
                cv.FONT_HERSHEY_PLAIN,
                2,
                color,
                2,
            )
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
