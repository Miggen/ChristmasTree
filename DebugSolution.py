import pickle
import matplotlib.pyplot as plt
from math import cos, sin, atan2
import numpy as np
import Lights
import cv2

focal_length_x = 645.0
focal_length_y = 634.0
center_x = 320.0
center_y = 240.0

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
    offset = 6 * cam_idx
    return offset

def residual(x, samples, first_light_idx):
    residual = []
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
        loss = (px[0] + ((focal_length_x * crcs[1]) / crcs[0]) - center_x)**2
        loss += (px[1] + ((focal_length_y * crcs[2]) / crcs[0]) - center_y)**2
        residual.append(loss)
    return residual

with open('/home/pi/Data/solution.pkl', 'rb') as f:
    result = pickle.load(f)
    samples = pickle.load(f)
    first_light_idx = pickle.load(f)

res = residual(result.x, samples, first_light_idx)
sort_index = np.argsort(res)
sort_index = np.flip(sort_index)
#plt.plot(res)
#plt.show()

#stop = False
#for i in sort_index:
#    light_idx, cam_idx, cam_id, px = samples[i]
#    img = cv2.imread(f'/home/pi/Data/SampleDebug/Debug_{light_idx:04d}_{cam_id:03d}.png')
#    try:
#        cv2.imshow('preview', img)
#    except:
#        continue
#    key = cv2.waitKey(0)
#
#    while True:
#        if key == 27: # exit on ESC
#            stop = True
#            break
#        elif key == 100:
#            print(f'({light_idx}, {cam_id}),')
#            break
#        elif key == 107:
#            break
#        key = cv2.waitKey(0)
#    if stop:
#        break

lights_x = []
lights_y = []
lights_z = []
for light_idx in range(0, Lights.NUM_LIGHTS - 1):
    light_offset = first_light_idx + (light_idx * 3)
    lights_x.append(result.x[light_offset])
    lights_y.append(result.x[light_offset + 1])
    lights_z.append(result.x[light_offset + 2])

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
