import cv2
from time import sleep


class Camera:
    def __init__(self, visualize):
        self._image_name = "preview"
        self.visualize = visualize
        cv2.namedWindow(self._image_name)
        self.stream = cv2.VideoCapture(0)
        if self.stream.isOpened():
            status, frame = self.stream.read()
            self.height = frame.shape[0]
            self.width = frame.shape[1]
        else:
            raise Exception("Unable to connect to camera")

    def __del__(self):
        self.stream.release()
        cv2.destroyAllWindows()

    def get(self):
        for _ in range(0, 6):
            status, frame = self.stream.read()
        if self.visualize:
            cv2.imshow(self._image_name, frame)
        return frame
