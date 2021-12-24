import argparse
from pathlib import Path
import pickle
import Lights
import numpy as np
import threading
import queue
from RotatingPlane import RotatingPlane
from Raindrops import Raindrops
from Pulses import Pulses


input_cmds = [
    "exit",
    "plane",
    "rain",
    "pulses",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Lights the tree in 3d patterns.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Data directory')
    return parser.parse_args()


def read_kbd_input(inputQueue):
    print('Ready for keyboard input:')
    while (True):
        input_str = input()
        inputQueue.put(input_str)


def load_light_positions(args):
    with open(f'{str(args.data_dir)}/solution.pkl', 'rb') as f:
        lights_3d = pickle.load(f)
    return lights_3d


def main():
    args = parse_arguments()
    inputQueue = queue.Queue()
    inputThread = threading.Thread(target=read_kbd_input, args=(inputQueue,), daemon=True)
    inputThread.start()

    if not args.data_dir.exists():
        raise Exception('Data directory is empty')

    lights_pos = load_light_positions(args)
    light_control = Lights.Lights()

    algo = RotatingPlane(lights_pos, light_control)

    keep_running = True
    while keep_running:
        try:
            algo.step()
            light_control.update()
            if inputQueue.qsize() > 0:
                input_str = inputQueue.get()
                if input_str == input_cmds[0]:
                    keep_running = False
                    break
                elif input_str == input_cmds[1]:
                    algo = RotatingPlane(lights_pos, light_control)
                elif input_str == input_cmds[2]:
                    algo = Raindrops(lights_pos, light_control)
                elif input_str == input_cmds[3]:
                    algo = Pulses(lights_pos, light_control)
                else:
                    print(f'Unknown command {input_str}')
                    print('Available commands:')
                    for cmd in input_cmds:
                        print(f'\t{cmd}')
        except Exception as e:
            print(e)
            keep_running = False


if __name__ == "__main__":
    main()
