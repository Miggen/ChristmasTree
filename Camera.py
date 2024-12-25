import cv2
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Camera:
    def __init__(self, manual_exposure=False):
        self.stream = cv2.VideoCapture(0)
        if manual_exposure:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.stream.set(cv2.CAP_PROP_EXPOSURE, 128)
        else:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

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
        for _ in range(5):
           status, frame = self.stream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb


if __name__ == "__main__":
    camera = Camera()
    initial_image = camera.get()
    fig, ax = plt.subplots()
    img = ax.imshow(initial_image)

    def _update_image(frame):
        image = camera.get()
        img.set_array(image)
        return [img]

    ani = animation.FuncAnimation(fig, _update_image, frames=None, interval=1000, blit=True)
    plt.show()
