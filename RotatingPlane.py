import numpy as np
import cv2


class RotatingPlane:
    def __init__(self, lights_pos, light_control):
        self.light_control = light_control
        self.lights_pos = lights_pos

        z_min = np.min(lights_pos[:, 2])
        z_max = np.max(lights_pos[:, 2])
        z_delta = z_max - z_min
        self.low_z = z_min + z_delta / 4.0
        self.high_z = z_max - z_delta / 2.0
        self.z_step = (self.high_z - self.low_z) / 300.0
        self.z = self.low_z
        self.increase_z = True

        rotation_axis = np.array([1.0 / 100.0, 0.0, 0.0])
        self.R = cv2.Rodrigues(rotation_axis)[0]
        point_on_plane = np.array([0.0, 0.0, self.z])
        self.plane = np.array([0.0, 0.0, 1.0, 0.0])
        self.plane[3] = np.sum(point_on_plane * self.plane[:3])

        self.color_1 = (127, 0, 255)
        self.color_2 = (255, 255, 0)

    def update_z(self):
        if self.increase_z:
            self.z += self.z_step
            if self.z + self.z_step > self.high_z:
                self.increase_z = False
        else:
            self.z -= self.z_step
            if self.z - self.z_step < self.low_z:
                self.increase_z = True

    def update_plane(self):
        self.plane[:3] = self.R @ self.plane[:3]
        point_on_plane = np.array([0.0, 0.0, self.z])
        self.plane[3] = np.sum(point_on_plane * self.plane[:3])

    def update_lights(self):
        for i in range(0, self.lights_pos.shape[0]):
            if np.sum(self.plane[:3] * self.lights_pos[i, :]) > self.plane[3]:
                color = self.color_1
            else:
                color = self.color_2
            self.light_control.set(i, color[0], color[1], color[2])

    def step(self):
        self.update_z()
        self.update_plane()
        self.update_lights()
