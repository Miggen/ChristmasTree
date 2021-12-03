def xmaslight():
    # This is the code from my 
    
    #NOTE THE LEDS ARE GRB COLOUR (NOT RGB)
    
    # Here are the libraries I am currently using:
    from time import sleep
    import board
    import neopixel
    import re
    import math
    
    # You are welcome to add any of these:
    # import random
    # import numpy
    # import scipy
    # import sys
    #set up the pixels (AKA 'LEDs')

    PIXEL_COUNT = 50 # this should be 500
    
    pixels = neopixel.NeoPixel(board.D21, PIXEL_COUNT, auto_write=False)
    pixels.fill((0, 0, 0))
    pixels.show()
    sleep(2)
    i = 0
    while True:
        for idx in range(0, PIXEL_COUNT):
            r = ((idx + i) * 5) % 255
            g = ((idx + i + 75) * 5) % 255
            b = ((idx + i + 150) * 5) % 255
            color = (g, r, b)
            pixels[idx] = color
        pixels.show()
        sleep(2)
        i += 1
        print('Done')

# yes, I just put this at the bottom so it auto runs
xmaslight()
