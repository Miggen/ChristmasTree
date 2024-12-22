import cv2
import numpy as np
import argparse
import Camera
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def calibrate_chessboard(images):
    '''Calibrate a camera using chessboard images.'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Iterate through all images
    for image_path in images:
        image_name = image_path.stem
        _, size_mm, num_cols, num_rows = image_name.split('_')
        size_cm = int(size_mm) / 10.0
        num_cols = int(num_cols)
        num_rows = int(num_rows)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
        objp = np.zeros((num_rows * num_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2)
        objp = objp * size_cm
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (num_cols, num_rows), None)
:w

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Computes camera calibration based on checkerboard pattern.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Data directory')
    parser.add_argument('--size-mm', required=True, type=int, help='Width/Height of checkerboard box  [mm]')
    parser.add_argument('--num-cols', required=True, type=int, help='Number of columns in checkerboard pattern')
    parser.add_argument('--num-rows', required=True, type=int, help='Number of rows in checkerboard pattern')
    parser.add_argument('--capture', action='store_true', help='Capture a new image')
    return parser.parse_args()


def main():
    args = parse_arguments()
    calibration_dir = args.data_dir / "Calibration"
    image_dir = calibration_dir / "Images"
    images = []
    if image_dir.exists():
        images = list(args.image_dir.glob('*.png'))

    if args.capture:
        camera = Camera.Camera()
        image = camera.get()
        calibration_dir.mkdir(exist_ok=True)
        image_dir.mkdir(exist_ok=True)
        image_path = image_dir / f"Checkerboard_{args.size_mm}_{args.num_cols}_{args.num_rows}.png"
        cv2.imwrite(image_path, image)
        images.append(image_path)
    else:
        ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(images)
        save_coefficients(mtx, dist, calibration_dir / "calibration.yml")


if __name__ == "__main__":
    main()
