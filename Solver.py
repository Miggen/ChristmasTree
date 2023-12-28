import argparse
import pickle
import Camera
import Lights
from scipy import optimize
from pathlib import Path
from math import cos, sin, atan2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb

PI = 3.1415

focal_length_x = 645.0
focal_length_y = 634.0
center_x = 320.0
center_y = 240.0

bad_samples = [
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Estimate the 3D positions of the Camera and Lights using samples from Calibration.py.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Data directory')
    return parser.parse_args()


def collect_samples(args):
    samples = []
    num_sample_sources = 0
    for sample_file_path in sorted(glob(f'{str(args.data_dir)}/sample_dmp_*.pkl')):
        with open(sample_file_path, 'rb') as sample_file:
            cam_samples = pickle.load(sample_file)
            for light_idx, cam_idx, px in cam_samples:
                if not (light_idx, cam_idx) in bad_samples:
                    samples.append((light_idx, num_sample_sources, cam_idx, px))
        num_sample_sources += 1
    return samples, num_sample_sources


def create_rotation(roll, pitch, yaw):
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


def get_rotation_jacobian(roll, pitch, yaw):
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    dr = np.zeros((3, 3))
    dr[0, 0] = 0.0
    dr[0, 1] = (cy * sp * cr) + (sy * sr)
    dr[0, 2] = -(cy * sp * sr) + (sy * cr)
    dr[1, 0] = 0.0
    dr[1, 1] = (sy * sp * cr) - (cy * sr)
    dr[1, 2] = -(sy * sp * sr) - (cy * cr)
    dr[2, 0] = 0.0
    dr[2, 1] = cp * cr
    dr[2, 2] = -cp * sr

    dp = np.zeros((3, 3))
    dp[0, 0] = -cy * sp
    dp[0, 1] = (cy * cp * sr)
    dp[0, 2] = (cy * cp * cr)
    dp[1, 0] = -sy * sp
    dp[1, 1] = (sy * cp * sr)
    dp[1, 2] = (sy * cp * cr)
    dp[2, 0] = -cp
    dp[2, 1] = -sp * sr
    dp[2, 2] = -sp * cr

    dy = np.zeros((3, 3))
    dy[0, 0] = -sy * cp
    dy[0, 1] = -(sy * sp * sr) - (cy * cr)
    dy[0, 2] = -(sy * sp * cr) + (cy * sr)
    dy[1, 0] = cy * cp
    dy[1, 1] = (cy * sp * sr) - (sy * cr)
    dy[1, 2] = (cy * sp * cr) + (sy * sr)
    dy[2, 0] = 0.0
    dy[2, 1] = 0.0
    dy[2, 2] = 0.0

    return dr, dp, dy


def tsc_to_crcs(to_crcs_R, to_tsc_t, tsc):
    crcs = to_crcs_R @ (tsc - to_tsc_t)
    return crcs


def crcs_to_tsc(to_tcs_R, to_tsc_t, crcs):
    tsc = (to_tcs_R @ crcs) + to_tsc_t
    return tsc


def get_cam_offset(cam_idx):
    offset = cam_idx * 6
    return offset


# sum (x + fx * vcrc / ucrc - cx)^2 + sum (y + fy * zcrc / ucrc - cy)^2
def loss_function(x, samples, first_light_idx):
    loss = 0.0
    prev_cam_idx = -1
    for light_idx, cam_idx, cam_id, px in samples:
        if prev_cam_idx != cam_idx:
            cam_offset = get_cam_offset(cam_idx)
            cam_u, cam_v, cam_z, cam_r, cam_p, cam_y = x[cam_offset:cam_offset + 6]

            to_tsc_R = create_rotation(cam_r, cam_p, cam_y)
            to_crcs_R = np.transpose(to_tsc_R)
            to_tsc_t = np.array([cam_u, cam_v, cam_z])
            prev_cam_idx = cam_idx

        light_offset = first_light_idx + (light_idx * 3)
        tsc = x[light_offset:light_offset + 3]

        crcs = tsc_to_crcs(to_crcs_R, to_tsc_t, tsc)
        loss += (px[0] + ((focal_length_x * crcs[1]) / crcs[0]) - center_x)**2
        loss += (px[1] + ((focal_length_y * crcs[2]) / crcs[0]) - center_y)**2
    print(f'{loss:e}')
    #visualize_state(x, first_light_idx)
    return loss


# g = sum (x + fx * vcrc / ucrc - cx)^2 + sum (y + fy * zcrc / ucrc - cy)^2
# g = h^2 + p^2, dg/da = 2 * h * dh/da + 2 * p * dp/da
def loss_jacobian(x, samples, first_light_idx):
    def jac_from_dcrcs(dcrcs, crcs, h, p):
        dcrcsxinv = dcrcs[0] / (crcs[0]**2)
        dcrcsy_crcsxinv = dcrcs[1] / crcs[0] + crcs[1] * dcrcsxinv
        dcrcsz_crcsxinv = dcrcs[2] / crcs[0] + crcs[2] * dcrcsxinv
        dh = focal_length_x * dcrcsy_crcsxinv
        dp = focal_length_y * dcrcsz_crcsxinv
        return  2.0 * h * dh + 2.0 * p * dp

    jac = np.zeros(len(x))
    jac_num_samples = np.ones(len(x))
    prev_cam_idx = -1
    for light_idx, cam_idx, cam_id, px in samples:
        if prev_cam_idx != cam_idx:
            cam_offset = get_cam_offset(cam_idx)
            cam_u, cam_v, cam_z, cam_r, cam_p, cam_y = x[cam_offset:cam_offset + 6]

            to_tsc_R = create_rotation(cam_r, cam_p, cam_y)
            to_crcs_R = np.transpose(to_tsc_R)
            to_tsc_t = np.array([cam_u, cam_v, cam_z])
            dtscRdroll, dtscRdpitch, dtscRdyaw = get_rotation_jacobian(cam_r, cam_p, cam_y)
            dcrcsRdroll = np.transpose(dtscRdroll)
            dcrcsRdpitch = np.transpose(dtscRdpitch)
            dcrcsRdyaw = np.transpose(dtscRdyaw)
            prev_cam_idx = cam_idx

        light_offset = first_light_idx + (light_idx * 3)
        tsc = x[light_offset:light_offset + 3]

        crcs = tsc_to_crcs(to_crcs_R, to_tsc_t, tsc)

        h = px[0] + ((focal_length_x * crcs[1]) / crcs[0]) - center_x
        p = px[1] + ((focal_length_y * crcs[2]) / crcs[0]) - center_y

        # dcamu
        dcrcs = -to_crcs_R[:, 0]
        jac[cam_offset] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[cam_offset] += 1.0

        # dcamv
        dcrcs = -to_crcs_R[:, 1]
        jac[cam_offset + 1] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[cam_offset + 1] += 1.0

        # dcamz
        dcrcs = -to_crcs_R[:, 2]
        jac[cam_offset + 2] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[cam_offset + 2] += 1.0

        # dcamr
        dcrcs = dcrcsRdroll @ (tsc - to_tsc_t)
        jac[cam_offset + 3] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[cam_offset + 3] += 1.0

        # dcamp
        dcrcs = dcrcsRdpitch @ (tsc - to_tsc_t)
        jac[cam_offset + 4] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[cam_offset + 4] += 1.0

        # dcamy
        dcrcs = dcrcsRdyaw @ (tsc - to_tsc_t)
        jac[cam_offset + 5] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[cam_offset + 5] += 1.0

        # dru
        dcrcs = to_crcs_R[:, 0]
        jac[light_offset] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[light_offset] += 1.0

        # drv
        dcrcs = to_crcs_R[:, 1]
        jac[light_offset + 1] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[light_offset + 1] += 1.0

        # drz
        dcrcs = to_crcs_R[:, 2]
        jac[light_offset + 2] += jac_from_dcrcs(dcrcs, crcs, h, p)
        jac_num_samples[light_offset + 2] += 1.0

    jac = jac / jac_num_samples
    return jac


def visualize_state(x, first_light_idx):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    num_cams_pos = (first_light_idx - 4 + 3) / 6.0
    X = []
    Y = []
    Z = []
    U = []
    V = []
    W = []
    for cam_idx in range(0, round(num_cams_pos)):
        cam_offset = get_cam_offset(cam_idx)
        cam_u, cam_v, cam_z, cam_r, cam_p, cam_y = x[cam_offset:cam_offset + 6]

        cam_tcs = np.array([cam_u, cam_v, cam_z])
        to_tsc_R = create_rotation(cam_r, cam_p, cam_y)
        u, v, w = crcs_to_tsc(to_tsc_R, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

        X.append(cam_u)
        Y.append(cam_v)
        Z.append(cam_z)
        U.append(u)
        V.append(v)
        W.append(w)

    ax.quiver(X, Y, Z, U, V, W)

    lights_x = []
    lights_y = []
    lights_z = []
    for light_idx in range(0, Lights.NUM_LIGHTS):
        light_offset = first_light_idx + (light_idx * 3)
        lights_x.append(x[light_offset])
        lights_y.append(x[light_offset + 1])
        lights_z.append(x[light_offset + 2])

    ax.scatter(lights_x, lights_y, lights_z)
    plt.show()


def main():
    args = parse_arguments()

    if not args.data_dir.exists():
        raise Exception('Data directory is empty')

    samples, num_sample_sources = collect_samples(args)
    if num_sample_sources == 0:
        raise Exception('No samples found')

    initial_cam_pos = [
        [0.1, -3., -1.0],
        [-2.0, 2.1, -1.0],
        [-3.1, 0.1, -2.0],
        [-1.0, 3.0, -2.1],
        [3.0, 3.1, -0.1],
        [2.0, -0.1, -2.0],
    ]

    initial_state = []
    for i in range(0, num_sample_sources):
        initial_state += initial_cam_pos[i] # u, v, z
        roll = 0.0
        pitch = 0.0
        x = initial_cam_pos[i][0]
        y = initial_cam_pos[i][1]
        yaw = PI + atan2(y, x)
        initial_state += [roll, pitch, yaw]

    first_light_idx = len(initial_state)
    # Last light position chosen as origin, with rotation from first camera
    for i in range(0, Lights.NUM_LIGHTS):
        initial_state += [0.0, 0.0, 0.0]

    result = optimize.minimize(loss_function,
                               x0=np.array(initial_state, dtype=np.float32),
                               jac=loss_jacobian,
                               args=(samples, first_light_idx),
                               method='Newton-CG')
    print(result)
    visualize_state(result.x, first_light_idx)
    with open(args.data_dir / f'solution.pkl', 'wb') as dmp_file:
        pickle.dump(result, dmp_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(samples, dmp_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(first_light_idx, dmp_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
