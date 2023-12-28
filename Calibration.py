import cv2
import argparse
import pickle
from Camera import Camera
import Lights
from time import sleep
from pathlib import Path
from glob import glob


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrate 3D positions of christmas lights.')
    parser.add_argument('--visualize', action='store_true', help='visualize camera image')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory')
    return parser.parse_args()


def identify_bright_spot(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    radius = 5
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return maxLoc, maxVal


def save_debug_img(rgb, maxLoc, maxVal, idx, camera_idx, args, debug_img_dir):
    cv2.circle(rgb, maxLoc, 5, (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb, f'{maxVal}',(10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    debug_img = str(debug_img_dir / f'Debug_{idx:04d}_{camera_idx:03d}.png')
    cv2.imwrite(debug_img, rgb)

    if args.visualize:
        cv2.imshow("preview", rgb)
        cv2.waitKey(1000)


def main():
    args = parse_arguments()
    args.output_dir.mkdir(exist_ok=True)
    camera = Camera(args.visualize)
    lights = Lights.Lights()

    sample_img_dir = args.output_dir / 'Sample'
    sample_img_dir.mkdir(exist_ok=True)

    debug_img_dir = args.output_dir / 'SampleDebug'
    debug_img_dir.mkdir(exist_ok=True)

    first_camera_idx = 0
    for filepath in glob(f'{str(args.output_dir)}/sample_dmp_*.pkl'):
        filename = Path(filepath).stem
        camera_idx = int(filename.split('_')[2])
        first_camera_idx = max(camera_idx + 1, first_camera_idx)

    for camera_idx in range(first_camera_idx, 1000):
        print('Move camera to next position, press ENTER to continue or ESC to stop')
        while True:
            rgb = camera.get()
            cv2.imshow("preview", rgb)
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                keep_running = False
                break
            elif key == 10 or key == 13:
                keep_running = True
                break

        if not keep_running:
            break

        baseline_rgb = camera.get()
        baseline_img = str(sample_img_dir / f'Baseline_{camera_idx:03d}.png')
        cv2.imwrite(baseline_img, baseline_rgb)

        samples = []
        for idx in range(0, Lights.NUM_LIGHTS):
            lights.set(idx, 255, 255, 255)
            lights.update()
            lights.set(idx, 0, 0, 0)
            rgb = camera.get()
            sample_img = str(sample_img_dir / f'Sample_{idx:04d}_{camera_idx:03d}.png')
            cv2.imwrite(sample_img, rgb)
            delta_rgb = cv2.subtract(rgb, baseline_rgb)
            maxLoc, maxVal = identify_bright_spot(delta_rgb)
            save_debug_img(rgb, maxLoc, maxVal, idx, camera_idx, args, debug_img_dir)

            if maxVal > 180:
                samples.append((idx, camera_idx, maxLoc))

        lights.update()

        with open(args.output_dir / f'sample_dmp_{camera_idx:03d}.pkl', 'wb') as dmp_file:
            pickle.dump(samples, dmp_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
