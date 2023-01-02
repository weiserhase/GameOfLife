import random
import string
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from numpy import all, array

import main as m


def id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


class CellState(Enum):
    DEAD = 0
    ALIVE = 1


def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


class Game:
    def __init__(self, size, id_size):
        self.size = size
        self.activeCells: list[np.ndarray] = []
        # Generate Neighbor Offsets and remove [0,0]
        lin = np.linspace(-1, 1, 3)
        self.mask = np.array(np.meshgrid(lin, lin),
                             dtype=int).ravel("F").reshape(-1, 2)
        print(self.mask)
        self.mask = np.delete(self.mask, 4, axis=0)
        print(self.mask)
        print("------------------")
        self.num_neigh = {}

    def change_cells(self, cells: list[tuple[np.ndarray, bool]]):
        for coordinates, state in cells:
            if (state == 1):
                self.activeCells.append(coordinates)

    def to_np_array(self) -> np.ndarray:
        arr = np.zeros((self.size, self.size), dtype=int)
        for coords in self.activeCells:
            try:
                arr[coords[0], coords[1]] = 1
            except Exception as e:
                print(coords, "Exception")
                raise Exception("Cannot convert")
        return arr

    def dict_np_array(self) -> np.ndarray:
        arr = np.zeros((self.size, self.size), dtype=int)
        for str_coords, num in self.num_neigh.items():
            coords = np.array(eval(str_coords), dtype=int)
            try:
                arr[coords[0], coords[1]] = num
            except Exception as e:
                print(coords, "Exception")
                raise Exception("Cannot convert")
        return arr

    def count_neighbors(self) -> dict[str, int]:
        timings = {}
        neighbors: dict[str, int] = {}
        # Fill a dict with the neighbor count
        timings["mask"] = []
        timings["repr"] = []
        timings["in"] = []
        timings["add"] = []
        for coord in self.activeCells:
            timings["mask"].append(time.time())
            neighbor_list = np.apply_along_axis(
                lambda x: x+coord, 1, self.mask)
            timings["mask"][-1] = time.time() - timings["mask"][-1]
            # neighbor_list = self.mask + coord
            for neighbor_coord in neighbor_list:
                # append new element or update new
                timings["repr"].append(time.time())
                str_coord = np.array_repr(neighbor_coord)
                timings["repr"][-1] = time.time() - timings["repr"][-1]
                timings["in"].append(time.time())
                if (str_coord not in neighbors):
                    neighbors[str_coord] = 1
                    timings["in"][-1] = time.time() - timings["in"][-1]
                    continue

                timings["in"][-1] = time.time() - timings["in"][-1]
                timings["add"].append(time.time())
                neighbors[str_coord] += 1
                timings["add"][-1] = time.time() - timings["add"][-1]

        new_timings = {}
        for key, l in timings.items():
            new_timings[key] = sum(l) / len(l)
        print(new_timings, "count timings", len(timings["mask"]))
        # self.num_neigh = neighbors
        return neighbors

    def cell_inbounds(self, coords):
        if (0 <= coords[0] < self.size):
            if (0 <= coords[1] < self.size):
                return True
        return False

    # def change_cell(self, coords):
    #     if (self.cell_inbounds(coords) == False):
    #         return
    #     if state == 0:
    #         return
    #     self.activeCells[str_coords] = 1

    def step(self):
        timings = {}
        timings["count"] = time.time()
        neighbors = self.count_neighbors()
        timings["count"] = time.time() - timings["count"]
        new_active = []
        for str_coord, num_neighbors in neighbors.items():
            # coords = np.frombuffer(str_coord, dtype=int)
            coords = np.array(eval(str_coord), dtype=int)
            # print(coords, str_coord)
            if (not self.cell_inbounds(coords)):
                continue
            if num_neighbors < 2:
                continue
            elif num_neighbors > 3:
                continue
            elif num_neighbors == 3:
                new_active.append(coords)
            else:
                if arreq_in_list(coords, self.activeCells):
                    new_active.append(coords)
                else:
                    continue
        print(timings, "step timings")
        self.activeCells = new_active
        return self.activeCells


def np_array2cells(grid):
    res = []
    grid_list = grid.tolist()
    for x, row in enumerate(grid_list):
        for y, value in enumerate(row):
            if (value == 1):
                res.append((np.array([x, y]), 1))
    return res


if __name__ == '__main__':
    grid = m.test_grid(2**4)
    normal_game = m.Game((2**8, 2**8), 2**1)
    normal_game.populate_grid(grid)
    # stepped = normal_game.step()

    # print(2**4-2**3)
    grid = np.pad(grid, (2**8 - 2**4)//2)
    game = Game(2**8, 0)
    game.change_cells(np_array2cells(grid))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(game.to_np_array(), cmap="gray")
    axs[0].set_title("Timon")
    axs[1].imshow(normal_game.grid, cmap="gray")
    axs[1].set_title("Jan")

    for i in range(10):
        t0 = time.time()
        game.step()
        print(time.time() - t0, "Timon")
        normal_game.generation()
        # print(game.activeCells)
        # print(np.vstack(
        #     np.where(normal_game.grid == 1)).ravel("F").reshape(-1, 2))

        if not np.array_equal(game.to_np_array(), normal_game.grid):
            print("inequal outputs")
        # raise Exception("Normal and Optimized outputs are not equal")
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(game.to_np_array(), cmap="gray")
        axs[0].set_title("Timon")
        axs[1].imshow(normal_game.grid, cmap="gray")
        axs[1].set_title("Jan")
        plt.show()
    # print(str(arr))
    # print(np.array(str(arr)))

    # mask = np.array(np.meshgrid(
    #     [-1, 1], [-1, 1])).ravel("F").reshape(-1, 2)
    # print(mask)
    # print(mask + np.array([1, 1]))
    print({1: 1, 2: 2} | {2: 3, 3: 4})
