import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pdb
from BundleAdjustment import visualize


def filter_bad_samples(n_cameras, camera_indices, point_indices, result):
    mapping = np.empty(n_cameras, dtype=int)
    mapping[0] = 0
    mapping[1] = 2
    mapping[2] = 3
    mapping[3] = 4
    mapping[4] = 5
    mapping[5] = 6
    mapping[6] = 8

    sort_index = np.argsort(np.abs(result.fun))
    sort_index = np.flip(sort_index)
    stop = False
    for i in sort_index:
        light_idx = point_indices[int(i / 2)]
        cam_idx = camera_indices[int(i / 2)]

        cam_id = mapping[cam_idx]
        img = cv2.imread(f'/home/pi/Data/SampleDebug/Debug_{light_idx:04d}_{cam_id:03d}.png')
        try:
            cv2.imshow('preview', img)
        except:
            continue
        key = cv2.waitKey(0)

        while True:
            if key == 27: # exit on ESC
                stop = True
                break
            elif key == 100:
                print(f'({light_idx}, {cam_id}),')
                break
            elif key == 107:
                break
            key = cv2.waitKey(0)
        if stop:
            break


with open('/home/pi/Data/solution_bundleAdjustment.pkl', 'rb') as f:
    result = pickle.load(f)
    n_cameras, n_points, camera_indices, point_indices, points_2d = pickle.load(f)


#filter_bad_samples(n_cameras, camera_indices, point_indices, result)
visualize(result.x, n_cameras, n_points)
#plt.plot(result.fun)
#plt.show()
