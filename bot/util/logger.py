import cv2 as cv


class Logger:

    # constants
    # Green (BGR)
    DEFAULT_TEXT_COLOR = (50, 255, 50)
    # Bottom-left corner of the text string in the image
    DEFAULT_TEXT_COORDINATES = (int(20), int(140))

    def logToImage(self, screenshot, text):
        self.logToImageWithColor(screenshot, text, self.DEFAULT_TEXT_COLOR)

    def logToImageWithColor(self, screenshot, text, color):
        self.logToImageWithColorAndCoordinates(
            screenshot, text, color, self.DEFAULT_TEXT_COORDINATES
        )

    def logToImageWithColorAndCoordinates(self, screenshot, text, color, coordinates):
        cv.putText(
            screenshot,
            text,
            coordinates,
            cv.FONT_HERSHEY_PLAIN,
            4,
            color,
            thickness=3,
        )
