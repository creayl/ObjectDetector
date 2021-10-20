from domain.point import Point


class BoundingBox:

    # properties
    pointStart = None
    pointEnd = None
    width = 0
    height = 0
    size = 0
    confidence = 0

    # constructor
    def __init__(self, x, y, xEnd, yEnd, confidence):
        self.pointStart = Point(x, y)
        self.pointEnd = Point(xEnd, yEnd)
        self.confidence = confidence
        self.height = self.pointEnd.y - self.pointStart.y
        self.width = self.pointEnd.x - self.pointStart.x
        self.size = self.height * self.width

    def score(self, screenSize):
        return (self.size / screenSize * 1.0) * (self.confidence * 1.0)
