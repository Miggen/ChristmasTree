import numpy as np
import cv2
from math import atan2

class RotatingTree:
    def __init__(self, lights_pos, light_control, speed_deg):
        if not speed_deg:
            self.speed_deg = 1.0
        else:
            self.speed_deg = float(speed_deg[0])

        self.colors = [
            (255, 0, 0),
            (0, 255, 0)]

        self.light_control = light_control
        self.lights_pos = lights_pos
        self.start_angle = 0.0
        self.z_min = np.min(lights_pos[:, 2])
        z_max = np.max(lights_pos[:, 2])
        self.z_delta = z_max - self.z_min

    def rotate(self):
        self.start_angle += np.deg2rad(self.speed_deg)
        if self.start_angle >= 2.0 * np.pi:
            self.start_angle -= 2.0 * np.pi

    def height_based_angle(self, z):
        rel_z = z - self.z_min
        num_rotations = 2.0
        rot_per_z = num_rotations * (np.pi * 2.0) / self.z_delta
        return rel_z * rot_per_z

    def normalize_angle(self, angle):
        twoPi = 2.0 * np.pi
        while angle >= twoPi:
            angle -= twoPi

        while angle < 0.0:
            angle += twoPi

        return angle

    def interpolate_color(self, remainder, color_1, color_2):
        a = (1.0 - remainder)
        b = remainder
        r = a * color_1[0] + b * color_2[0]
        g = a * color_1[1] + b * color_2[1]
        b = a * color_1[2] + b * color_2[2]
        return (int(r), int(g), int(b))

    def color_from_angle(self, angle):
        angle = self.normalize_angle(angle)
        color_idx = (angle / (2.0 * np.pi)) * (len(self.colors) - 1)
        low_idx = int(color_idx)
        color_1 = self.colors[low_idx]
        color_2 = self.colors[low_idx + 1]
        remainder = color_idx - low_idx
        return self.interpolate_color(remainder, color_1, color_2)

    def update_lights(self):
        for i in range(0, self.lights_pos.shape[0]):
            pos = self.lights_pos[i, :]
            angle = atan2(pos[1], pos[0])
            angle += self.start_angle
            angle += self.height_based_angle(pos[2])
            color = self.color_from_angle(angle)
            self.light_control.set(i, color[0], color[1], color[2])
        self.light_control.update()

    def step(self):
        self.rotate()
        self.update_lights()
