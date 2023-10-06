import math
from time import process_time

class Moment:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.time = process_time()

    def __str__(self):
        return f"Point at ({self.x}, {self.y}) at {self.time}"

    def angleBetween(self, middlePoint, endPoint) -> int:
        a = [self.x, self.y]
        b = [middlePoint.x, middlePoint.y]
        c = [endPoint.x, endPoint.y]
        
        radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        angle = abs(radians*180.0/math.pi)

        if angle >180.0:
            angle = 360-angle
            
        return angle
