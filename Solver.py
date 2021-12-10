import argparse
import pickle
import Camera
import Lights
from scipy import optimize
from pathlib import Path
from math import cos, sin
import numpy as np
from glob import glob

import pdb


def parse_arguments():
    parser = argparse.ArgumentParser(description='Estimate the 3D positions of the Camera and Lights using samples from Calibration.py.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Data directory')
    return parser.parse_args()


def collect_samples(args):
    samples = []
    num_sample_sources = 0
    for sample_file_path in glob(f'{str(args.data_dir)}/sample_dmp_*.pkl'):
        with open(sample_file_path, 'rb') as sample_file:
            cam_samples = pickle.load(sample_file)
            for light_idx, cam_idx, px in cam_samples:
                samples.append((light_idx, num_sample_sources, px))
        num_sample_sources += 1
    return samples, num_sample_sources


def create_transform(u, v, z, roll, pitch, yaw):
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    T = np.zeros((4, 4))
    T[0, 0] = cy * cp
    T[0, 1] = (cy * sp * sr) - (sy * cr)
    T[0, 2] = (cy * sp * cr) + (sy * sr)
    T[0, 3] = u
    T[1, 0] = sy * cp
    T[1, 1] = (sy * sp * sr) + (cy * cr)
    T[1, 2] = (sy * sp * cr) - (cy * sr)
    T[1, 3] = v
    T[2, 0] = -sp
    T[2, 1] = cp * sr
    T[2, 2] = cp * cr
    T[2, 3] = z
    T[3, 3] = 1.0

    return T


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


def transform(T, u, v, z):
    res = T @ np.transpose(np.array([u, v, z, 1.0]))
    return res[0], res[1], res[2]


def get_cam_offset(cam_idx):
    offset = 4
    if cam_idx > 0:
        offset += 3 +  (6 * (cam_idx - 1))
    return offset


# sum (x + fx * vcrc / ucrc - cx)^2 + sum (y + fy * zcrc / ucrc - cy)^2
def loss_function(x, samples, first_light_idx):
    fx, fy, cx, cy = x[0:4]

    loss = 0.0
    prev_cam_idx = -1
    for light_idx, cam_idx, px in samples:
        if prev_cam_idx != cam_idx:
            cam_offset = get_cam_offset(cam_idx)
            if cam_idx > 0:
                cam_u, cam_v, cam_z, cam_r, cam_p, cam_y = x[cam_offset:cam_offset + 6]
            else:
                cam_u, cam_v, cam_z = x[cam_offset:cam_offset + 3]
                cam_r, cam_p, cam_y = [0.0, 0.0, 0.0]

            if abs(cam_u) < 0.01:
                pdb.set_trace()
            T = create_transform(cam_u, cam_v, cam_z, cam_r, cam_p, cam_y)
            prev_cam_idx = cam_idx

        if light_idx < (Lights.NUM_LIGHTS - 1):
            light_offset = first_light_idx + (light_idx * 3)
            ru, rv, rz = x[light_offset:light_offset + 3]
        else:
            ru, rv, rz = [0.0, 0.0, 0.0]

        lu, lv, lz = transform(T, ru, rv, rz)
        loss += (px[0] + ((fx * lv) / lu) - cx)**2
        loss += (px[1] + ((fy * lz) / lu) - cy)**2
    print(loss)
    return loss


# g = sum (x + fx * vcrc / ucrc - cx)^2 + sum (y + fy * zcrc / ucrc - cy)^2
# g = h^2 + p^2, dg/da = 2 * h * dh/da + 2 * p * dp/da
def loss_jacobian(x, samples, first_light_idx):
    fx, fy, cx, cy = x[0:4]
    jac = np.zeros(len(x))
    prev_cam_idx = -1
    for light_idx, cam_idx, px in samples:
        if prev_cam_idx != cam_idx:
            cam_offset = get_cam_offset(cam_idx)
            if cam_idx > 0:
                cam_u, cam_v, cam_z, cam_r, cam_p, cam_y = x[cam_offset:cam_offset + 6]
            else:
                cam_u, cam_v, cam_z = x[cam_offset:cam_offset + 3]
                cam_r, cam_p, cam_y = [0.0, 0.0, 0.0]

            T = create_transform(cam_u, cam_v, cam_z, cam_r, cam_p, cam_y)
            dTdroll, dTdpitch, dTdyaw = get_rotation_jacobian(cam_r, cam_p, cam_y)
            prev_cam_idx = cam_idx

        if light_idx < (Lights.NUM_LIGHTS - 1):
            light_offset = first_light_idx + (light_idx * 3)
            ru, rv, rz = x[light_offset:light_offset + 3]
        else:
            ru, rv, rz = [0.0, 0.0, 0.0]

        lu, lv, lz = transform(T, ru, rv, rz)

        h = px[0] + ((fx * lv) / lu) - cx
        p = px[1] + ((fy * lz) / lu) - cy

        # dfx
        jac[0] += 2 * h * lv / lu

        # dfy
        jac[1] += 2 * p * lz / lu

        # dcx
        jac[2] += -2 * h

        # dcy
        jac[3] += -2 * p

        # dcamu
        dlu = 1.0
        dluinv = -dlu / (lu**2)
        dh = fx * lv * dluinv
        dp = fy * lz * dluinv
        jac[cam_offset] += 2 * h * dh + 2 * p * dp

        # dcamv
        dlv = 1.0
        dh = (fx / lu) * dlv
        jac[cam_offset + 1] += 2 * h * dh

        # dcamz
        dlz = 1.0
        dp = (fy / lu) * dlz
        jac[cam_offset + 2] += 2 * p * dp

        if cam_idx > 0:
            ruvz = np.array([ru, rv, rz])
            # dcamr
            du, dv, dz = dTdroll @ ruvz
            duinv = -du / (lu**2)
            dvuinv = dv / lu + lv * duinv
            dzuinv = dz / lu + lz * duinv
            dh = fx * dvuinv
            dp = fy * dzuinv
            jac[cam_offset + 3] += 2 * h * dh + 2 * p * dp

            # dcamp
            du, dv, dz = dTdpitch @ ruvz
            duinv = -du / (lu**2)
            dvuinv = dv / lu + lv * duinv
            dzuinv = dz / lu + lz * duinv
            dh = fx * dvuinv
            dp = fy * dzuinv
            jac[cam_offset + 4] += 2 * h * dh + 2 * p * dp

            # dcamy
            du, dv, dz = dTdyaw @ ruvz
            duinv = -du / (lu**2)
            dvuinv = dv / lu + lv * duinv
            dzuinv = dz / lu + lz * duinv
            dh = fx * dvuinv
            dp = fy * dzuinv
            jac[cam_offset + 5] += 2 * h * dh + 2 * p * dp

        if light_idx < (Lights.NUM_LIGHTS - 1):
            # dru
            du, dv, dz = T[0:3, 0]
            duinv = -du / (lu**2)
            dvuinv = dv / lu + lv * duinv
            dzuinv = dz / lu + lz * duinv
            dh = fx * dvuinv
            dp = fy * dzuinv
            jac[light_offset] += 2 * h * dh + 2 * p * dp

            # drv
            du, dv, dz = T[0:3, 1]
            duinv = -du / (lu**2)
            dvuinv = dv / lu + lv * duinv
            dzuinv = dz / lu + lz * duinv
            dh = fx * dvuinv
            dp = fy * dzuinv
            jac[light_offset + 1] += 2 * h * dh + 2 * p * dp

            # drz
            du, dv, dz = T[0:3, 2]
            duinv = -du / (lu**2)
            dvuinv = dv / lu + lv * duinv
            dzuinv = dz / lu + lz * duinv
            dh = fx * dvuinv
            dp = fy * dzuinv
            jac[light_offset + 2] += 2 * h * dh + 2 * p * dp
    return jac


def main():
    args = parse_arguments()

    if not args.data_dir.exists():
        raise Exception('Data directory is empty')

    samples, num_sample_sources = collect_samples(args)
    if num_sample_sources == 0:
        raise Exception('No samples found')

    focal_length_x = 100.0
    focal_length_y = 100.0
    center_x = 240.0
    center_y = 240.0

    initial_state = [focal_length_x, focal_length_y, center_x, center_y]
    initial_state += [-10.0, 0.0, 0.0] # u, v, z (angles 0 by definition)
    for i in range(0, num_sample_sources - 1):
        initial_state += [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0] # u, v, z, roll, pitch, yaw

    first_light_idx = len(initial_state)
    # Last light position chosen as origin, with rotation from first camera
    for i in range(0, Lights.NUM_LIGHTS - 1):
        initial_state += [0.0, 0.0, 0.0]

    result = optimize.minimize(loss_function,
                               x0=np.array(initial_state, dtype=np.float32),
                               jac=loss_jacobian,
                               args=(samples, first_light_idx),
                               method='Newton-CG')
    print(result)


if __name__ == "__main__":
    main()
