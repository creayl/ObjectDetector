import cv2 as cv
import tensorflow as tf
from typing import List

from domain.boundingbox import BoundingBox


class AI:
    # constants
    

    # properties
    detect_fn = None

    def __init__(self, detect_fn):
        self.detect_fn = detect_fn
    

    def findTrees(self, screenshot):
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
        detections = self.detect_fn(input_tensor)
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
        return bboxList