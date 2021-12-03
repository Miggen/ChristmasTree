import cv2
import argparse
from Camera import Camera
from Lights import Lights
from time import sleep


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrate 3D positions of christmas lights.')
    parser.add_argument('--visualize', action='store_true', help='visualize camera image')
    return parser.parse_args()


def identify_bright_spot(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    radius = 5
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return maxLoc


def main():
    args = parse_arguments()
    camera = Camera(args.visualize)
    lights = Lights()

    for idx in range(0, lights.num_lights):
        lights.set(idx, 255, 255, 255)
        lights.update()
        lights.set(idx, 0, 0, 0)
        rgb = camera.get()
        maxLoc = identify_bright_spot(rgb)
        cv2.circle(rgb, maxLoc, 5, (255, 0, 0), 2)
        cv2.imshow("preview", rgb)
        cv2.waitKey(1000)

if __name__ == "__main__":
    main()
