import numpy as np
import random
from scipy.spatial import distance
from math import sqrt, acos, atan2


snake_color = (255, 255, 255)
food_color = (255, 0, 0)
no_color = (0, 0, 0)


class Snake:
    def __init__(self, lights_pos, light_control):
        self.light_control = light_control
        self.lights_pos = lights_pos
        n_lights = lights_pos.shape[0]
        self.neighbors = np.zeros((n_lights, 6), dtype=int) - 1
        for src in range(0, n_lights):
            r_min = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
            for dst in range(src + 1, n_lights):
                delta = lights_pos[dst] - lights_pos[src]
                delta_sum = np.sum(np.abs(delta))
                if delta_sum < 0.3:
                    sectors = self.sector_from_delta(*delta)
                    for sector_idx in sectors:
                        if delta_sum < r_min[sector_idx]:
                            r_min[sector_idx] = delta_sum
                            self.neighbors[src, sector_idx] = dst
                            self.neighbors[dst, self.invert_sector(sector_idx)] = src
        self.reset()

    def invert_sector(self, sector):
        if sector == 0:
            inv_sector = 2
        elif sector == 1:
            inv_sector = 3
        elif sector == 2:
            inv_sector = 0
        elif sector == 3:
            inv_sector = 1
        elif sector == 4:
            inv_sector = 5
        elif sector == 5:
            inv_sector = 4
        else:
            inv_sector = -1
        return inv_sector

    def sector_from_delta(self, x, y, z):
        sectors = []
        if x > 0.0:
            sectors.append(0)
        else:
            sectors.append(2)

        if y > 0.0:
            sectors.append(1)
        else:
            sectors.append(3)

        if z > 0.0:
            sectors.append(4)
        else:
            sectors.append(5)

        pos = [abs(x), abs(y), abs(z)]
        sectors = [s for _, s in sorted(zip(pos, sectors), reverse=True)]
        return sectors

    def reset(self):
        self.light_control.set_all(*no_color)
        self.snake = []
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
        curr_idx = self.snake[0]
        delta = self.lights_pos[self.food, :] - self.lights_pos[curr_idx, :]
        delta_sum = np.sum(np.abs(delta))
        sectors = self.sector_from_delta(*delta)
        print(f'Curr: {curr_idx} {self.lights_pos[curr_idx, :]}')
        print(f'Food: {self.food} {self.lights_pos[self.food, :]}')
        for i, n in enumerate(self.neighbors[curr_idx, :]):
            print(f'Neighbor: {i} {n} {self.lights_pos[n, :]}')
        for sector in sectors:
            neighbor = self.neighbors[curr_idx, sector]
            if neighbor >= 0 and neighbor not in self.snake:
                neighbor_delta_sum = np.sum(np.abs(self.lights_pos[self.food, :] - self.lights_pos[neighbor, :]))
                if neighbor_delta_sum < delta_sum:
                    print(f'Result: {neighbor}')
                    return neighbor
        return self.snake[0]

    def move_snake(self):
        next_idx = self.next_snake_idx()
        if next_idx in self.snake:
            self.reset()
        else:
            self.snake.insert(0, next_idx)
            self.light_control.set(next_idx, *snake_color)
            if next_idx == self.food:
                self.create_food()
            else:
                outgoing_idx = self.snake.pop()
                self.light_control.set(outgoing_idx, *no_color)

    def step(self):
        self.move_snake()
