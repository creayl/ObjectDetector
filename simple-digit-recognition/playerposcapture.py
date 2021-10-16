import win32gui, win32ui, win32con
import numpy as np


class PlayerPositionCapture:
    
    def get_screenshot(_self, x, y, width, height):
        # get the window image data
        # grab a handle to the main desktop window
        hwnd = win32gui.GetDesktopWindow()
        # create a device context
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        # create a memory based device context
        cDC = dcObj.CreateCompatibleDC()
        # create a bitmap object
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(dcObj, width, height)
        cDC.SelectObject(screenshot)
        # copy the screen into our memory device context
        cDC.BitBlt((0, 0), (width, height), dcObj, (x, y), win32con.SRCCOPY)

        # output screenshot for debbuging purpose
        screenshot.SaveBitmapFile(cDC, "debug.bmp")

        # convert the raw data into a format opencv can read
        signedIntsArray = screenshot.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype="uint8")
        img.shape = (height, width, 4)

        # free resources
        cDC.DeleteDC()
        dcObj.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(screenshot.GetHandle())
        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[..., :3]
        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)
        return img
