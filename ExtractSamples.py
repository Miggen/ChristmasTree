import cv2
import argparse
import pickle
import Lights
from pathlib import Path
import SampleLightPositions
import matplotlib.pyplot as plt
from enum import Enum
import math
import yaml
import numpy as np


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
    def __init__(self, bits=None, x_sum=0.0, y_sum=0.0, num_position_samples=0):
        if bits == None:
            self.bits = [BitValue.NOT_SET] * SampleLightPositions.NUM_BITS_NEEDED
        else:
            self.bits = bits
        self.x_sum = x_sum
        self.y_sum = y_sum
        self.num_position_samples = num_position_samples

    def match(self, image_point):
        x, y = self.position()
        distance = math.sqrt((x - image_point["x"])**2 + (y - image_point["y"])**2)
        threshold = 2.0
        return distance < threshold

    def update(self, image_point, bit, bit_value: BitValue):
        self.x_sum += image_point["x"]
        self.y_sum += image_point["y"]
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
        return f"Sample(bits={self.bits!r}, x_sum={self.x_sum!r} y_sum={self.y_sum!r}, num_position_samples={self.num_position_samples!r})"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrate 3D positions of christmas lights.')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory')
    return parser.parse_args()


def identify_bright_dots(image):
    bright_dots = []

    _, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            # Calculate the centroid (center)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            mask = np.zeros(image.shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(image, mask=mask)[0]

            bright_dots.append({
                "x": cX, 
                "y": cY, 
                "intensity": round(mean_val, 2),
                "size": cv2.contourArea(contour)
            })

    return bright_dots


def remove_background(image, background):
    return cv2.subtract(image, background)


def extract_bright_points(images, debug_dir: Path):
    bright_points = dict()
    background_rgb = cv2.imread(images["background"])
    background = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY)
    for image_name, image_path in images.items():
        if "bit" in image_name:
            image_rgb = cv2.imread(image_path)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            image = remove_background(image, background)
            cv2.imwrite(debug_dir / f"{image_name}_removed_background.png", image)
            bright_dots = identify_bright_dots(image)
            bright_points[image_name] = bright_dots

            for dot in bright_dots:
                area = dot["size"]
                radius = np.sqrt(area / np.pi)
                cv2.circle(image_rgb, (int(dot["x"]), int(dot["y"])), int(radius), (255, 0, 0), 1)

            cv2.imwrite(debug_dir / f"{image_name}_bright_points.png", image_rgb)

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
        cv2.putText(image, str(idx), (int(x) - 10, int(y) + 5), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_dir / "samples.png", image)


def save_data(output_dir, samples):
    with open(output_dir / "sample.pkl", 'wb') as file:
        pickle.dump(samples, file)


def load_data(samples_dir):
    data = dict()
    for sample_dir in samples_dir.iterdir():
        sample_file = sample_dir / "sample.pkl"
        if sample_file.exists():
            with open(sample_file, 'rb') as file:
                camera_position, sample_data = pickle.load(file)
                data[sample_dir.stem] = (camera_position, sample_data)

    return data

def get_camera_position(sample_dir):
    camera_position_path = sample_dir / "camera_position.yaml"
    with open(camera_position_path, 'r') as file:
        camera_position = yaml.safe_load(file)

    return camera_position


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

        images_points = extract_bright_points(images, debug_dir)
        light_samples = match_bright_points(images_points)
        samples = filter_samples(light_samples)
        draw_samples(debug_dir, samples, images)
        camera_position = get_camera_position(sample_dir)
        save_data(sample_dir, (camera_position, samples))


if __name__ == "__main__":
    main()
