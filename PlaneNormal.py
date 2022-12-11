import numpy as np
import cv2
from random import random

class PlaneNormal:
    def __init__(self, lights_pos, light_control, normal):
        try:
            self.normal = np.array([float(normal[0]), float(normal[1]), float(normal[2])])
            self.normal *= 1.0 / np.linalg.norm(self.normal)
        except:
            print(f"Invalid plane normal, got {normal}, expected 3 floats, defaulting to 1.0 0.0 0.0")
            self.normal = np.array([1.0, 0.0, 0.0])

        self.light_control = light_control
        self.lights_pos = lights_pos

        self.update_color()

        self.min_c = 100000.0
        self.max_c = -100000.0
        for i in range(0, self.lights_pos.shape[0]):
            point = self.lights_pos[i, :]
            c = self.point_eq(point)
            self.min_c = min(self.min_c, c)
            self.max_c = max(self.max_c, c)

        self.c = self.min_c

    def update_color(self):
        self.color = np.random.choice(range(256), size=3)
        while sum(self.color) < 50:
            self.color = np.random.choice(range(256), size=3)

    def point_eq(self, point):
        return np.sum(self.normal[:] * point)

    def update_c(self):
        step_length = 0.05
        self.c += step_length
        if self.c > self.max_c + step_length:
            self.c = self.min_c
            self.update_color()

    def update_lights(self):
        self.light_control.dim_all(0.8)
        for i in range(0, self.lights_pos.shape[0]):
            c = self.point_eq(self.lights_pos[i, :])
            if c < self.c and c > self.c - 0.2:
                self.light_control.set(i, self.color[0], self.color[1], self.color[2])
        self.light_control.update()

    def step(self):
        self.update_c()
        self.update_lights()
