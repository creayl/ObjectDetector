import pyautogui
import cv2 as cv2
import numpy as np
import os
import random
import win32api, win32con
from numpy.testing._private.nosetester import NoseTester
from time import time

from util.windowcapture import WindowCapture
from util.vision import Vision
from util.logger import Logger
from util.utils import Utils

# This file is for testing code parts only.. dont remove useless imports in this file please. ;)
# i just wanna throw in code parts here and be able to run and test them fast


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# properties
pyautogui.FAILSAFE = False
is_autowalk = False
should_end = False
canHarvestVision: Vision = Vision("img/harvest_tooltip.png")
stillHarvestingVision: Vision = Vision("img/harvest_in_progress.png")
wincap: WindowCapture = WindowCapture(None)
logger: Logger = Logger()
utils: Utils = Utils()

#load image with alpha channel.  use IMREAD_UNCHANGED to ensure loading of alpha channel
image = cv2.imread('your image', cv2.IMREAD_UNCHANGED)    

#make mask of where the transparent bits are
trans_mask = image[:,:,3] == 0

#replace areas of transparency with white and not transparent
image[trans_mask] = [255, 255, 255, 255]

#new image without alpha channel...
new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)