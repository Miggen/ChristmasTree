import numpy as np
import imageio


class GifVisualizer:
    def __init__(self, lights_pos, light_control, gif_path, color_scale):
        self.light_control = light_control
        self.lights_pos = lights_pos
        self.color_scale = color_scale
        self.pos_min = np.min(self.lights_pos, axis=0)
        self.pos_max = np.max(self.lights_pos, axis=0)
        gif = imageio.get_reader(gif_path)
        self.current_frame = 0

        self.images = []
        for image in gif:
            self.images.append(image)
        self.yz_proj = np.rint(self.scaled_projection([0.0, 0.0, 100000.0,
                                                       3.0, 0.0, -(self.pos_min[2] + self.pos_max[2]) / 2.0]))

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs)
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def project(self, points, camera_pos):
        """Convert 3-D points to 2-D by projecting onto images."""
        normalized_proj = self.rotate(points, camera_pos[:3])
        normalized_proj += camera_pos[3:6]
        normalized_proj = -normalized_proj[:, 1:] / normalized_proj[:, 0, np.newaxis]
        return normalized_proj

    def scaled_projection(self, camera_pos):
        normalized_proj = self.project(self.lights_pos, camera_pos)
        min_proj = np.min(normalized_proj, axis=0)
        max_proj = np.max(normalized_proj, axis=0)
        size = self.images[0].shape
        width = size[1] - 1
        height = size[0] - 1
        x_scale = width / (max_proj[0] - min_proj[0])
        x_bias = -x_scale * min_proj[0]
        y_scale = height / (max_proj[1] - min_proj[1])
        y_bias = -y_scale * min_proj[1]
        return (normalized_proj * [x_scale, y_scale]) + [x_bias, y_bias]

    def update_image(self):
        self.current_frame += 1
        if self.current_frame >= len(self.images):
            self.current_frame = 0

    def update_lights(self):
        image = self.images[self.current_frame]
        for i, px in enumerate(self.yz_proj):
            rgb = image[int(px[1]), int(px[0])]
            rgb = np.array(rgb[:3], dtype=float)
            rgb *= self.color_scale
            rgb = rgb.astype(int)
            self.light_control.set(i, rgb[0], rgb[1], rgb[2])

    def step(self):
        self.update_image()
        self.update_lights()
