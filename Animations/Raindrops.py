import numpy as np
import cv2
from random import random
from math import cos, sin


class Raindrops:
    def __init__(self, lights_pos, light_control):
        self.light_control = light_control
        self.lights_pos = lights_pos
        self.drops = []
        self.color = (100, 100, 255)
        self.z_max = np.max(self.lights_pos[:, 2])
        self.z_min = np.min(self.lights_pos[:, 2])
        self.light_control.set_all(0, 0, 0)

    def create_drop(self):
        angle = 6.28 * random()
        radius = random()
        x = radius * cos(angle)
        y = radius * sin(angle)
        self.drops.append([x, y, self.z_max])

    def update_drops(self):
        for i in range(len(self.drops) - 1, -1, -1):
            self.drops[i][2] -= 0.02
            if self.drops[i][2] < self.z_min:
                self.drops.pop(i)

    def update_lights(self):
        self.light_control.dim_all(0.95)
        limit = 0.1**2
        for drop in self.drops:
            dist = np.sum(np.square(self.lights_pos - drop), axis=1)
            close_points = np.where(dist < limit)[0]
            for idx in close_points:
                self.light_control.set(idx, self.color[0], self.color[1], self.color[2])

    def step(self):
        if random() > 0.9:
            self.create_drop()

        self.update_drops()
        self.update_lights()
