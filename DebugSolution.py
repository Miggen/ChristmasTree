import pickle
import matplotlib.pyplot as plt
from math import cos, sin, atan2
import numpy as np
import Lights

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


def tsc_to_crcs(to_crcs_R, to_tsc_t, tsc):
    crcs = to_crcs_R @ (tsc - to_tsc_t)
    return crcs


def crcs_to_tsc(to_tcs_R, to_tsc_t, crcs):
    tsc = (to_tcs_R @ crcs) + to_tsc_t
    return tsc


def get_cam_offset(cam_idx):
    offset = 4
    if cam_idx > 0:
        offset += 3 +  (6 * (cam_idx - 1))
    return offset

def residual(x, samples, first_light_idx):
    fx, fy, cx, cy = x[0:4]

    residual = []
    prev_cam_idx = -1
    for light_idx, cam_idx, px in samples:
        if prev_cam_idx != cam_idx:
            cam_offset = get_cam_offset(cam_idx)
            if cam_idx > 0:
                cam_u, cam_v, cam_z, cam_r, cam_p, cam_y = x[cam_offset:cam_offset + 6]
            else:
                cam_u, cam_v, cam_z = x[cam_offset:cam_offset + 3]
                cam_r, cam_p, cam_y = [0.0, 0.0, 0.0]

            to_tsc_R = create_rotation(cam_r, cam_p, cam_y)
            to_crcs_R = np.transpose(to_tsc_R)
            to_tsc_t = np.array([cam_u, cam_v, cam_z])
            prev_cam_idx = cam_idx

        if light_idx < (Lights.NUM_LIGHTS - 1):
            light_offset = first_light_idx + (light_idx * 3)
            tsc = x[light_offset:light_offset + 3]
        else:
            tsc = np.array([0.0, 0.0, 0.0])
        crcs = tsc_to_crcs(to_crcs_R, to_tsc_t, tsc)
        loss = (px[0] + ((fx * crcs[1]) / crcs[0]) - cx)**2
        loss += (px[1] + ((fy * crcs[2]) / crcs[0]) - cy)**2
        residual.append(loss)
    return residual

with open('/home/pi/Data/solution.pkl', 'rb') as f:
    result = pickle.load(f)
    samples = pickle.load(f)
    first_light_idx = pickle.load(f)


lights_x = []
lights_y = []
lights_z = []
for light_idx in range(0, Lights.NUM_LIGHTS - 1):
    light_offset = first_light_idx + (light_idx * 3)
    lights_x.append(result.x[light_offset])
    lights_y.append(result.x[light_offset + 1])
    lights_z.append(result.x[light_offset + 2])

lights_x.append(0.0)
lights_y.append(0.0)
lights_z.append(0.0)

lights_x.reverse()
lights_y.reverse()
lights_z.reverse()

prev_x = 0.0
prev_y = 0.0
prev_z = 0.0
deltas = []
for x, y, z in zip(lights_x, lights_y, lights_z):
    dist = ((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)**(0.5)
    deltas.append(dist)
    prev_x = x
    prev_y = y
    prev_z = z

plt.plot(deltas)
plt.show()
