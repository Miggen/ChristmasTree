import board
from neopixel import NeoPixel


class Lights:
    def __init__(self):
        self.num_lights = 50
        self.pixels = NeoPixel(board.D21, self.num_lights, auto_write=False)
        self.set_all(0, 0, 0)

    def __del__(self):
        self.set_all(0, 0, 0)

    def set_all(self, r, g, b):
        color = (min(g, 255), min(r, 255), min(b, 255))
        self.pixels.fill(color)
        self.pixels.show()

    def set(self, idx, r, g, b):
        color = (min(g, 255), min(r, 255), min(b, 255))
        if idx >= self.num_lights:
            raise Exception(f"Light index out of range {idx} / {self.num_lights}")
        self.pixels[idx] = color

    def update(self):
        self.pixels.show()
