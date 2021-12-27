import numpy as np
import random
from scipy.spatial import distance
from math import sqrt, acos, atan2
import time


snake_color = (255, 255, 255)
food_color = (255, 0, 0)
no_color = (0, 0, 0)


class Snake:
    def __init__(self, lights_pos, light_control):
        self.light_control = light_control
        self.lights_pos = lights_pos
        n_lights = lights_pos.shape[0]
        self.distances = distance.squareform(distance.pdist(lights_pos, 'cityblock'))
        closest = np.argsort(self.distances, axis=1)
        self.neighbors = []
        for src in range(0, n_lights):
            neighbors = []
            for dst in closest[src, 1:]:
                if self.distances[src, dst] > 0.3:
                    break

                closer_neighbor = False
                for neighbor in neighbors:
                    if self.distances[neighbor, dst] < self.distances[src, dst]:
                        closer_neighbor = True
                        break

                if not closer_neighbor:
                    neighbors.append(dst)

            self.neighbors.append(neighbors)

        self.reset()
        self.last_time = time.time()

    def reset(self):
        self.light_control.set_all(*no_color)
        self.snake = []
        self.history = []
        self.snake.append(self.get_new_idx())
        self.light_control.set(self.snake[0], *snake_color)
        self.create_food()

    def create_food(self):
        idx = self.get_new_idx()
        self.food = idx
        self.light_control.set(idx, *food_color)

    def get_new_idx(self):
        available_indices = list(range(0, self.lights_pos.shape[0]))
        for i in self.snake:
            available_indices.remove(i)
        return random.choice(available_indices)

    def next_snake_idx(self):
        src = self.snake[0]
        next_idx = src
        best_dist = 10000.0
        for dst in self.neighbors[src]:
            if dst not in self.snake and dst not in self.history:
                dist = self.distances[dst, self.food]
                if dist < best_dist:
                    best_dist = dist
                    next_idx = dst
        return next_idx

    def move_snake(self):
        next_idx = self.next_snake_idx()
        if next_idx in self.snake:
            self.reset()
        else:
            self.snake.insert(0, next_idx)
            self.light_control.set(next_idx, *snake_color)
            if next_idx == self.food:
                self.create_food()
                self.history = []
            else:
                outgoing_idx = self.snake.pop()
                self.light_control.set(outgoing_idx, *no_color)
                self.history.append(outgoing_idx)

    def step(self):
        self.move_snake()
        elapsed_time = time.time() - self.last_time
        remaining_time = 0.1 - elapsed_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)
        self.last_time = time.time()

