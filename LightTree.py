import argparse
from pathlib import Path
import pickle
import Lights
import numpy as np
import threading
import queue
import os
from RotatingPlane import RotatingPlane
from Raindrops import Raindrops
from Pulses import Pulses
from GifVisualizer import GifVisualizer
from Snake import Snake
from PlaneNormal import PlaneNormal
from RotatingTree import RotatingTree


input_cmds = [
    "exit",
    "plane",
    "rain",
    "pulses",
    "fire",
    "snake",
    "normal",
    "rotating",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Lights the tree in 3d patterns.')
    parser.add_argument('--data-dir', required=True, type=Path, help='Data directory')
    parser.add_argument('--mode', required=False, type=str, help='Initial mode to run', default='rotating')
    parser.add_argument('--no-io', action='store_true', help='Disables reading of input')
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


def select_algo(mode, mode_args, lights_pos, light_control, prev_algo):
    algo = prev_algo
    if mode == input_cmds[0]:
        algo = None
        pass
    elif mode == input_cmds[1]:
        algo = RotatingPlane(lights_pos, light_control)
    elif mode == input_cmds[2]:
        algo = Raindrops(lights_pos, light_control)
    elif mode == input_cmds[3]:
        algo = Pulses(lights_pos, light_control, mode_args)
    elif mode == input_cmds[4]:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        algo = GifVisualizer(lights_pos, light_control,
                f'{dir_path}/Animated_fire_by_nevit.gif', [1.0, 0.5, 0.5])
    elif mode == input_cmds[5]:
        algo = Snake(lights_pos, light_control)
    elif mode == input_cmds[6]:
        algo = PlaneNormal(lights_pos, light_control, mode_args)
    elif mode == input_cmds[7]:
        algo = RotatingTree(lights_pos, light_control, mode_args)
    else:
        if not mode == "help":
            print(f'Unknown command {mode}')
        print('Available commands:')
        for cmd in input_cmds:
            print(f'\t{cmd}')
    return algo


def main():
    args = parse_arguments()
    inputQueue = queue.Queue()
    if not args.no_io:
        inputThread = threading.Thread(target=read_kbd_input, args=(inputQueue,), daemon=True)
        inputThread.start()

    if not args.data_dir.exists():
        raise Exception('Data directory is empty')

    lights_pos = load_light_positions(args)
    light_control = Lights.Lights()

    algo = select_algo(args.mode, [], lights_pos, light_control, None)

    while algo is not None:
        try:
            algo.step()
            light_control.update()
            if inputQueue.qsize() > 0:
                input_str = inputQueue.get().split(" ")
                mode = input_str[0]
                mode_args = input_str[1:]
                algo = select_algo(mode, mode_args, lights_pos, light_control, algo)
        except Exception as e:
            print(e)
            algo = None


if __name__ == "__main__":
    main()
