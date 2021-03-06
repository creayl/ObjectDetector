import sys
import numpy as np
import cv2
import os

# Change the working directory to the folder this script is in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = cv2.imread('data/debug.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 3, 2)

#################      Now finding Contours         ###################
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100), np.float32)
responses = []
keys = [i for i in range(48, 58)]

for cnt in contours:
    [x, y, w, h] = cv2.boundingRect(cnt)
    if h > 7 and w*h > 100 and w*h < 200:
        print("w*h: ", w*h)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = thresh[y:y + h, x:x + w]
        roismall = cv2.resize(roi, (10, 10))
        cv2.imshow('norm', im)
        key = cv2.waitKey(0)

        if key == 27:  # (escape to quit)
            sys.exit()
        elif key in keys:
            responses.append(int(chr(key)))
            sample = roismall.reshape((1, 100))
            samples = np.append(samples, sample, 0)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

samples = np.float32(samples)
responses = np.float32(responses)

cv2.imwrite("data/train_result.png", im)
np.savetxt('data/generalsamples.data', samples)
np.savetxt('data/generalresponses.data', responses)