import cv2
from time import sleep
import matplotlib.pyplot as plt


class Camera:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.stream.set(cv2.CAP_PROP_EXPOSURE, 128)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if self.stream.isOpened():
            status, frame = self.stream.read()
            self.height = frame.shape[0]
            self.width = frame.shape[1]
        else:
            raise Exception("Unable to connect to camera")

    def __del__(self):
        self.stream.release()

    def get(self):
        for _ in range(0, 6):
            status, frame = self.stream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


if __name__ == "__main__":
    camera = Camera()
    image = camera.get()
    plt.imshow(image)
    plt.title("Camera Image")
    plt.axis("off")
    plt.show()
