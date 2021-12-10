import board
from neopixel import NeoPixel


NUM_LIGHTS = 500


class Lights:
    def __init__(self):
        self.pixels = NeoPixel(board.D21, NUM_LIGHTS, auto_write=False)
        self.set_all(0, 0, 0)

    def __del__(self):
        self.set_all(0, 0, 0)

    def set_all(self, r, g, b):
        color = (min(g, 255), min(r, 255), min(b, 255))
        self.pixels.fill(color)
        self.pixels.show()

    def set(self, idx, r, g, b):
        color = (min(g, 255), min(r, 255), min(b, 255))
        if idx >= NUM_LIGHTS:
            raise Exception(f"Light index out of range {idx} / {NUM_LIGHTS}")
        self.pixels[idx] = color

    def update(self):
        self.pixels.show()
