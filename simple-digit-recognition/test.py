import cv2
import numpy as np
import os
from playerposcapture import PlayerPositionCapture


# Change the working directory to the folder this script is in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the PlayerPositionCapture class
playerposcap = PlayerPositionCapture()

x = 2300
y = 20
width = 248
height = 16
#im = playerposcap.get_screenshot(x, y, width, height)

#######   training part    ###############
samples = np.loadtxt("data/generalsamples.data", np.float32)
responses = np.loadtxt("data/generalresponses.data", np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################
im = cv2.imread("data/debug.png")
out = np.zeros(im.shape, np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 3, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

lastX = 0
i = 0
numbers = []

for cnt in contours:
    [x, y, w, h] = cv2.boundingRect(cnt)
    if h > 7 and w*h > 100 and w*h < 200:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = thresh[y : y + h, x : x + w]
        roismall = cv2.resize(roi, (10, 10))
        roismall = roismall.reshape((1, 100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
        string = str(int((results[0][0])))

      
        print("string, x, lastX", string, x, lastX)

        if lastX and lastX - x  > 15:
            numbers.reverse()
            print(numbers)
            numbers.clear()


        numbers.append(string)  
        lastX = x
        cv2.putText(out, string, (x, y + h), 0, 0.5, (0, 255, 0))

cv2.imshow("im", im)
cv2.imshow("out", out)
cv2.waitKey(0)
