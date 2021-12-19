import argparse
import pickle
import cv2
from pathlib import Path
from glob import glob
import Lights
import numpy as np
from math import cos, sin, atan2
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares

import pdb

PI = 3.1415
focal_length_x = 645.0
focal_length_y = 634.0
center_x = 320.0
center_y = 240.0

bad_samples = [
    (187, 5),
    (303, 5),
    (209, 5),
    (57, 3),
    (305, 5),
    (58, 3),
    (60, 3),
    (115, 5),
    (199, 3),
    (196, 0),
    (399, 3),
    (198, 3),
    (60, 4),
    (397, 3),
    (190, 3),
    (193, 3),
    (266, 5),
    (192, 3),
    (74, 3),
    (345, 5),
    (196, 3),
    (53, 2),
    (171, 3),
    (325, 5),
    (55, 3),
    (51, 5),
    (61, 3),
    (35, 8),
    (89, 0),
    (89, 6),
    (370, 5)
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Estimate the 3D positions of the Camera and Lights using samples from Calibration.py.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Data directory')
    return parser.parse_args()

def rotation_matrix_from_euler(roll, pitch, yaw):
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    R = np.zeros((3, 3))
    R[0, 0] = cy * cp
    R[0, 1] = (cy * sp * sr) - (sy * cr)
    R[0, 2] = (cy * sp * cr) + (sy * sr)
    R[1, 0] = sy * cp
    R[1, 1] = (sy * sp * sr) + (cy * cr)
    R[1, 2] = (sy * sp * cr) - (cy * sr)
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr

    return R

def initialize(args):
    data_files = sorted(glob(f'{str(args.data_dir)}/sample_dmp_*.pkl'))
    n_cameras = len(data_files)
    n_points = Lights.NUM_LIGHTS

    samples = []
    num_sample_sources = 0
    for sample_file_path in data_files:
        with open(sample_file_path, 'rb') as sample_file:
            cam_samples = pickle.load(sample_file)
            for light_idx, cam_idx, px in cam_samples:
                if not (light_idx, cam_idx) in bad_samples:
                    samples.append((light_idx, num_sample_sources, cam_idx, px))
        num_sample_sources += 1

    n_observations = len(samples)
    camera_indices = np.empty(n_observations, dtype=int)
    point_indices = np.empty(n_observations, dtype=int)
    points_2d = np.empty((n_observations, 2), dtype=float)

    for i, sample in enumerate(samples):
        camera_indices[i] = sample[1]
        point_indices[i] = sample[0]
        points_2d[i] = [float(sample[3][0] - center_x), float(sample[3][1] - center_y)]

    initial_cam_pos = [
        [-3.0, 0.1, -2.0],
        [3.0, 0.1, -2.0],
        [2.0, 3.0, -1.0],
        [-1.0, 3.0, -0.1],
        [3.0, 0.1, -1.0],
        [-3.0, -0.1, -1.0],
        [0.1, -3.0, -2.0],
    ]

    camera_params = np.empty((n_cameras, 6), dtype=float)
    for i in range(n_cameras):
        x, y, z = initial_cam_pos[i]
        roll = 0.0
        pitch = 0.0
        yaw = PI + atan2(y, x)
        R = rotation_matrix_from_euler(roll, pitch, yaw)
        rodrigues = cv2.Rodrigues(R)

        camera_params[i, 0:3] = rodrigues[0].reshape(3)
        camera_params[i, 3:6] = initial_cam_pos[i]

    points_3d = np.zeros((n_points, 3), dtype=float)

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = focal_length_x
    k1 = 0.0
    k2 = 0.0
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


num_calls = 0
def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    global num_calls
    if num_calls % 12 == 0:
        visualize(params, n_cameras, n_points)
    num_calls += 1
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def visualize(x, n_cameras, n_points):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    lights_offset = n_cameras * 6

    lights_3d = x[lights_offset:].reshape((n_points, 3))

    cam_pos = np.empty((n_cameras, 6), dtype=float)
    for i in range(0, n_cameras):
        offset = i * 6
        cam_pos[i, 0:3] = x[i+3:i+6]
        R = np.empty((3,3), dtype=float)
        cv2.Rodrigues(x[i:i+3], R)
        cam_pos[i, 3:6] = R @ np.array([1.0, 0.0, 0.0])

    ax.quiver(cam_pos[:,0], cam_pos[:,1], cam_pos[:,2], cam_pos[:,3], cam_pos[:,4], cam_pos[:,5], color='r')
    ax.scatter(lights_3d[:,0], lights_3d[:, 1], lights_3d[:, 2])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plt.show()

def main():
    args = parse_arguments()

    if not args.data_dir.exists():
        raise Exception('Data directory is empty')

    camera_params, points_3d, camera_indices, point_indices, points_2d = initialize(args)
    if len(points_2d) == 0:
        raise Exception('No samples found')
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', loss='soft_l1',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    with open(args.data_dir / f'solution_bundleAdjustment.pkl', 'wb') as dmp_file:
        pickle.dump(res, dmp_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump((n_cameras, n_points, camera_indices, point_indices, points_2d), dmp_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
