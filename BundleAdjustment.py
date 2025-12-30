import argparse
import pickle
import cv2
from pathlib import Path
import Lights
import numpy as np
from math import cos, sin, atan2
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares
from CalibrateCamera import load_coefficients
from ExtractSamples import load_data, Sample, BitValue


def parse_arguments():
    parser = argparse.ArgumentParser(description='Estimate the 3D positions of the Camera and Lights using samples from ExtractSamples.py.')
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

def initialize(data_dir):
    samples_dir = data_dir / "Raw_Samples"
    samples_data = load_data(samples_dir)
    n_cameras = len(samples_data)
    n_points = Lights.NUM_LIGHTS

    camera_positions = dict()
    samples = []
    for cam_idx, sample_name in enumerate(samples_data.keys()):
        camera_position, frame_samples = samples_data[sample_name]
        camera_positions[cam_idx] = camera_position
        for sample in frame_samples:
            x, y = sample.position()
            light_indices = sample.index()
            for light_idx in light_indices:
                # Any idx above number of target lights is incorrect
                if light_idx < Lights.NUM_LIGHTS:
                    samples.append((cam_idx, light_idx, x, y))

    camera_matrix, distortion_coeff = load_coefficients(data_dir)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    n_observations = len(samples)
    camera_indices = np.empty(n_observations, dtype=int)
    point_indices = np.empty(n_observations, dtype=int)
    normalized_raw = np.empty((n_observations, 2), dtype=float)

    for i, sample in enumerate(samples):
        cam_idx, light_idx, x_raw, y_raw = sample
        camera_indices[i] = cam_idx
        point_indices[i] = light_idx
        x = (float(x_raw) - cx) / fx
        y = (float(y_raw) - cy) / fy
        normalized_raw[i] = [x, y]
    #normalized_warped = cv2.undistortPoints(normalized_raw, camera_matrix, distortion_coeff).reshape(-1, 2)


    camera_pos = np.empty((n_cameras, 6), dtype=float)
    for cam_idx in range(n_cameras):
        position = camera_positions[cam_idx]
        x = position['position']['x']
        y = position['position']['y']
        z = position['position']['z']
        target_rotation_deg = position['orientation']['yaw']
        target_rotation = np.deg2rad(target_rotation_deg)
        roll = 0.0
        pitch = 0.0
        yaw = np.pi + atan2(y, x) + target_rotation
        R = rotation_matrix_from_euler(-roll, -pitch, -yaw)
        rodrigues = cv2.Rodrigues(R)

        camera_pos[cam_idx, :3] = rodrigues[0].reshape(3)
        camera_pos[cam_idx, 3:] = -R @ [x, y, z]

    points_3d = np.zeros((n_points, 3), dtype=float)

    return camera_pos, points_3d, camera_indices, point_indices, normalized_raw

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

def project(points, camera_pos):
    """Convert 3-D points to 2-D by projecting onto images."""
    normalized_proj = rotate(points, camera_pos[:, :3])
    normalized_proj += camera_pos[:, 3:6]
    normalized_proj = -normalized_proj[:, 1:] / normalized_proj[:, 0, np.newaxis]
    return normalized_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, normalized_warped):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_pos = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    normalized_proj = project(points_3d[point_indices], camera_pos[camera_indices])
    projection_residual = (normalized_proj - normalized_warped).ravel()

    expected_max_distance = 0.15
    point_distances = np.linalg.norm(points_3d[:-1, :] - points_3d[1:, :], axis=1)
    distance_residual = np.maximum(0.0, point_distances - expected_max_distance)

    return np.concatenate([projection_residual, distance_residual])

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2 + (n_points - 1)
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    j = np.arange(n_points - 1)
    for s in range(3 * 2):
        A[camera_indices.size * 2 + j, n_cameras * 6 + j * 3 + s] = 1

    return A

def visualize(params, n_cameras, n_points):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    lights_offset = n_cameras * 6

    lights_3d = params[lights_offset:].reshape((n_points, 3))

    camera_pos = params[:n_cameras * 6].reshape((n_cameras, 6))
    camera_arrows = np.empty((n_cameras, 6), dtype=float)
    for i in range(0, n_cameras):
        R = np.empty((3,3), dtype=float)
        cv2.Rodrigues(camera_pos[i, :3], R)
        camera_arrows[i, :3] = np.transpose(-R) @ camera_pos[i, 3:]
        camera_arrows[i, 3:] = np.transpose(R) @ np.array([0.1, 0.0, 0.0])

    ax.quiver(camera_arrows[:,0], camera_arrows[:,1], camera_arrows[:,2], camera_arrows[:,3], camera_arrows[:,4], camera_arrows[:,5], color='r')
    ax.scatter(lights_3d[:,0], lights_3d[:, 1], lights_3d[:, 2])
    plt.show()

def get_points(params, n_cameras, n_points):
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_3d -= points_3d[-1, :]
    mean_point = np.mean(points_3d, axis=0)
    points_3d = np.transpose(points_3d)

    w = mean_point / np.linalg.norm(mean_point)
    u = np.array([1.0, 0.0, -w[0] / w[2]])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)

    R = [-u, -v, -w]

    points_3d = R @ points_3d
    points_3d = np.transpose(points_3d)
    return points_3d

def main():
    args = parse_arguments()

    if not args.data_dir.exists():
        raise Exception('Data directory is empty')

    camera_pos, points_3d, camera_indices, point_indices, normalized_warped = initialize(args.data_dir)
    if len(normalized_warped) == 0:
        raise Exception('No samples found')
    n_cameras = camera_pos.shape[0]
    n_points = points_3d.shape[0]

    x0 = np.hstack((camera_pos.ravel(), points_3d.ravel()))

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', loss='soft_l1',
                        args=(n_cameras, n_points, camera_indices, point_indices, normalized_warped))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    with open(args.data_dir / f'solution_bundleAdjustment_dbg.pkl', '+wb') as dmp_file:
        pickle.dump(res, dmp_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump((n_cameras, n_points, camera_indices, point_indices, normalized_warped), dmp_file, pickle.HIGHEST_PROTOCOL)

    rotated_points = get_points(res.x, n_cameras, n_points)
    #visualize(res.x, n_cameras, n_points)
    with open(args.data_dir / f'solution.pkl', 'wb') as dmp_file:
        pickle.dump(rotated_points, dmp_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
