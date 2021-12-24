import numpy as np
from random import random
from math import cos, sin


class Pulses:
    def __init__(self, lights_pos, light_control):
        self.light_control = light_control
        self.lights_pos = lights_pos
        self.pos_min = np.min(self.lights_pos, axis=0)
        self.pos_max = np.max(self.lights_pos, axis=0)
        self.center = (self.pos_min + self.pos_max) / 2.0
        self.max_radius = np.max(np.abs(lights_pos - self.center))
        self.radius = 1000.0
        self.color = [0, 0, 0]
        self.step_length = 0.05

    def update_pulse(self):
        if self.radius > self.max_radius:
            self.radius = 0.0
            self.color = np.random.choice(range(256), size=3)
            for i in range(0, 3):
                self.center[i] = self.pos_min[i] + random() * (self.pos_max[i] - self.pos_min[i])
            self.max_radius = np.max(np.abs(self.lights_pos - self.center))
        self.radius += self.step_length

    def update_lights(self):
        self.light_control.dim_all(0.85)
        limit = (self.radius + self.step_length)**2 - self.radius**2
        point_radius_squared = np.sum(np.square(self.lights_pos - self.center), axis=1)
        close_points = np.where(np.abs(point_radius_squared - self.radius**2) < limit)[0]
        for idx in close_points:
            self.light_control.set(idx, self.color[0], self.color[1], self.color[2])

    def step(self):
        self.update_pulse()
        self.update_lights()
