import cv2
import argparse
from Camera import Camera
import Lights
from time import sleep
from pathlib import Path
from math import log, ceil


NUM_BITS_NEEDED = ceil(log(Lights.NUM_LIGHTS) / log(2))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect images to calibrate 3D positions of christmas lights.')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory')
    return parser.parse_args()


def save_image(camera, output_dir, filename):
    sleep(1.0)
    rgb = camera.get_rgb()
    sample_file = output_dir / filename
    cv2.imwrite(str(sample_file), rgb)


def collect_samples(camera, lights, output_dir):
    lights.set_all(0, 0, 0)
    save_image(camera, output_dir, f"Sample_background.png")

    lights.set_all(255, 255, 255)
    save_image(camera, output_dir, f"Sample_all.png")

    for bit in range(0, NUM_BITS_NEEDED):
        for bit_value in range(0, 2):
            for light_idx in range(0, Lights.NUM_LIGHTS):
                is_bit_set = ((light_idx >> bit) & 1) == bit_value
                if is_bit_set:
                    lights.set(light_idx, 255, 255, 255)
                else:
                    lights.set(light_idx, 0, 0, 0)

            lights.update()
            save_image(camera, output_dir, f"Sample_bit_{bit}_{bit_value}.png")


def main():
    args = parse_arguments()
    camera = Camera(manual_exposure=True)
    lights = Lights.Lights()

    sample_img_dir = args.output_dir / 'Raw_Samples'
    sample_img_dir.mkdir(exist_ok=True)

    sample_number = 0
    for folder in sample_img_dir.iterdir():
        folder_number = int(folder.stem.split('_')[-1])
        sample_number = max(sample_number, folder_number + 1)

    image_dir = sample_img_dir / f'Sample_{sample_number}'
    image_dir.mkdir()

    print('Move camera to next position, press ENTER to continue or ESC to stop')
    lights.set_all(255, 255, 255)
    while True:
        rgb = camera.get_rgb()
        display_img = cv2.resize(rgb, (800, 540))
        cv2.imshow("preview", display_img)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            return
        elif key == 10 or key == 13:
            break

    collect_samples(camera, lights, image_dir)


if __name__ == "__main__":
    main()
