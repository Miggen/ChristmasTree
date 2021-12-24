import board
from neopixel import NeoPixel


NUM_LIGHTS = 500


class Lights:
    def __init__(self):
        self.pixels = NeoPixel(board.D21, NUM_LIGHTS, auto_write=False)
        self.set_all(0, 0, 0)

    def __del__(self):
        self.set_all(0, 0, 0)

    def set_all(self, r, g, b, update=True):
        color = (min(g, 255), min(r, 255), min(b, 255))
        self.pixels.fill(color)
        if update:
            self.pixels.show()

    def set(self, idx, r, g, b):
        color = (min(g, 255), min(r, 255), min(b, 255))
        if idx >= NUM_LIGHTS:
            raise Exception(f"Light index out of range {idx} / {NUM_LIGHTS}")
        self.pixels[idx] = color

    def dim_all(self, factor):
        for i in range(0, NUM_LIGHTS):
            color = self.pixels[i]
            r = float(color[1])
            g = float(color[0])
            b = float(color[2])
            r *= factor
            g *= factor
            b *= factor
            r = int(r)
            g = int(g)
            b = int(b)
            self.set(i, r, g, b)

    def update(self):
        self.pixels.show()
