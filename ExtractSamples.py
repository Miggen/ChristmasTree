import cv2
import argparse
import pickle
import Lights
from pathlib import Path
import SampleLightPositions
import matplotlib.pyplot as plt
from enum import Enum
import math


class BitValue(Enum):
    ZERO = 0
    ONE = 1
    BOTH = 2
    NOT_SET = 3

    @staticmethod
    def from_string(label: str):
        if label == '0':
            return BitValue.ZERO
        elif label == '1':
            return BitValue.ONE
        else:
            raise ValueError(f"'{label}' is not a valid BitValue")

class Sample:
    def __init__(self):
        self.bits = [BitValue.NOT_SET] * SampleLightPositions.NUM_BITS_NEEDED
        self.x_sum = 0.0
        self.y_sum = 0.0
        self.num_position_samples = 0

    def match(self, image_point):
        x, y = self.position()
        point_x, point_y, _ = image_point
        distance = math.sqrt((x - point_x)**2 + (y - point_y)**2)
        threshold = 2.0
        return distance < threshold

    def update(self, image_point, bit, bit_value: BitValue):
        point_x, point_y, _ = image_point
        self.x_sum += point_x
        self.y_sum += point_y
        self.num_position_samples += 1

        if self.bits[bit] == BitValue.NOT_SET:
            self.bits[bit] = bit_value
        elif self.bits[bit] != bit_value:
            self.bits[bit] = BitValue.BOTH

    def position(self):
        if self.num_position_samples > 0:
            x = self.x_sum / self.num_position_samples
            y = self.y_sum / self.num_position_samples
        else:
            x = -1000.0
            y = -1000.0

        return x, y

    def index(self):
        values = [0]
        for bit_position, bit in enumerate(self.bits):
            num_candidates = len(values)
            if bit == BitValue.ONE:
                for i in range(num_candidates):
                    values[i] += 2**bit_position
            elif bit == BitValue.BOTH:
                for i in range(num_candidates):
                    # Duplicate all candidates
                    values.append(values[i])
                    values[-1] += 2**bit_position
            elif bit == BitValue.NOT_SET:
                values = []
                break
        return values


    def __str__(self):
        x, y = self.position()
        ind = self.index()
        bit_string = ""
        for bit in reversed(self.bits):
            bit_string += f"{bit.value}"

        return f"{x}, {y}: {ind} {bit_string}"

    def __repr__(self):
        return self.__str__()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrate 3D positions of christmas lights.')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory')
    return parser.parse_args()


def identify_bright_dots(image):
    bright_dots = []
    _, thresholded = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calculate moments of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            # Calculate the centroid (center) of the contour
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            intensity = image[cY, cX]
            bright_dots.append((cX, cY, intensity))
    return bright_dots


def save_debug_img(rgb, maxLoc, maxVal, idx, camera_idx, args, debug_img_dir):
    cv2.circle(rgb, maxLoc, 5, (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb, f'{maxVal}',(10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    debug_img = str(debug_img_dir / f'Debug_{idx:04d}_{camera_idx:03d}.png')
    cv2.imwrite(debug_img, rgb)

    if args.visualize:
        cv2.imshow("preview", rgb)
        cv2.waitKey(1000)


def remove_background(image, background):
    return cv2.subtract(image, background)


def extract_bright_points(images):
    bright_points = dict()
    background_rgb = cv2.imread(images["background"])
    background = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY)
    for image_name, image_path in images.items():
        if "bit" in image_name:
            image_rgb = cv2.imread(image_path)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            image = remove_background(image, background)
            bright_dots = identify_bright_dots(image)
            bright_points[image_name] = bright_dots
    return bright_points


def match_bright_points(images_points):
    samples = []
    for image_name, image_points in images_points.items():
        _, bit, bit_value = image_name.split('_')
        bit = int(bit)
        bit_value = BitValue.from_string(bit_value)

        for image_point in image_points:
            match_found = False
            for sample in samples:
                if sample.match(image_point):
                    sample.update(image_point, bit, bit_value)
                    match_found = True

            if not match_found:
                new_sample = Sample()
                new_sample.update(image_point, bit, bit_value)
                samples.append(new_sample)

    return samples

def filter_samples(raw_samples):
    filtered_samples = []
    for sample in raw_samples:
        if len(sample.index()) > 0:
            filtered_samples.append(sample)
    return filtered_samples

def draw_samples(output_dir, samples, images):
    image = cv2.imread(images['all'])
    for sample in samples:
        x, y = sample.position()
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        idx = sample.index()
        if len(idx) == 1:
            idx = idx[0]
        cv2.putText(image, str(idx), (int(x) - 10, int(y) + 5), font, 0.25, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_dir / "samples.png", image)


def main():
    args = parse_arguments()

    samples_dir = args.output_dir / "Raw_Samples"

    num_expected_images = SampleLightPositions.NUM_BITS_NEEDED * 2 + 2 # 2 per bit + all + background
    for sample_dir in samples_dir.iterdir():
        images = dict()
        for image_path in sample_dir.glob("*.png"):
            image_name_parts = image_path.stem.split("_")
            image_name = "_".join(image_name_parts[1:])
            images[image_name] = image_path

        if len(images.keys()) != num_expected_images:
            print(f"Unexpected number of images found in {sample_dir.stem} found {len(images.keys())} expected {num_expected_images}, skipping sample")
            continue

        debug_dir = sample_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

        images_points = extract_bright_points(images)
        light_samples = match_bright_points(images_points)
        samples = filter_samples(light_samples)
        draw_samples(debug_dir, samples, images)


if __name__ == "__main__":
    main()
