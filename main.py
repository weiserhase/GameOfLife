import multiprocessing
import time
from enum import Enum

# import numba as nb
import numpy as np
from matplotlib import pyplot as plt


class CellState(Enum):
    DEAD = 0
    ALIVE = 1


def count_neighbors(matrix) -> int:
    matrix[matrix.shape[0]//2, matrix.shape[1]//2] = 0
    return np.count_nonzero(matrix.ravel())


def new_state(matrix) -> CellState:
    num_neighbors = count_neighbors(np.copy(matrix))
    if num_neighbors == 3:
        return CellState.ALIVE
    if num_neighbors < 2:
        return CellState.DEAD
    if num_neighbors > 3:
        return CellState.DEAD
    return CellState(matrix[matrix.shape[0]//2, matrix.shape[1]//2])


# @nb.njit(fastmath=True)
def get_partial_matrix(grid: np.ndarray, coordinates: tuple[int, int], neighbor_size: int = 1):
    return grid[coordinates[0]:coordinates[0]+neighbor_size*2+1,
                coordinates[1]:coordinates[1]+neighbor_size*2+1]


# Disecting  grid
"""
A grid can be disected into n pieces
"""


def change_neighbors(grid: np.ndarray):
    """ All elements where a neighbor contains cells are set to 1"""
    res = np.array(np.where(grid == 1)).ravel("F").reshape(-1, 2)
    for coord in res:
        grid[coord[0]-1:coord[0]+2,
             coord[1]-1:coord[1]+2] = 1
    return grid


def find_smallest_divisor(shape):
    for div in range(2, min(shape)):
        if (shape[0] % div == 0 and shape[1] % div == 0):
            return div
    return 2


def disect_grid(grid, size_th=2**0):
    if (grid.shape[0] <= size_th or grid.shape[1] <= size_th):
        if (np.count_nonzero(grid) == 0):
            return np.array([0])
        else:
            return np.array([1])

    new_grid = np.zeros(
        (grid.shape[0]//size_th, grid.shape[0]//size_th), dtype=int)
    # print(grid)
    # TODO Find divisors instead of using of using linspace (this enables any grid size)
    div = find_smallest_divisor(grid.shape)
    lin = np.arange(div)
    offsets = np.array(np.meshgrid(lin, lin)).ravel(order="F").reshape(-1, 2)

    for offset in offsets:
        x_pad = (offset[0]*grid.shape[0]//2,
                 (offset[0]+1)*grid.shape[0]//2)
        y_pad = (offset[1]*grid.shape[1]//2,
                 (offset[1]+1)*grid.shape[1]//2)

        partial_grid = grid[x_pad[0]: x_pad[1], y_pad[0]: y_pad[1]]
        sum = np.count_nonzero(partial_grid)

        if (sum != 0):
            x_offset = (offset[0]*new_grid.shape[0]//2,
                        (offset[0]+1)*new_grid.shape[0]//2)
            y_offset = (offset[1]*new_grid.shape[1]//2,
                        (offset[1]+1)*new_grid.shape[1]//2)
            n_g = disect_grid(partial_grid, size_th)
            new_grid[x_offset[0]: x_offset[1], y_offset[0]: y_offset[1]] = n_g

    return new_grid


def increase_grid(grid, size):
    fact = size[0]//grid.shape[0], size[1]//grid.shape[1]
    arr = np.repeat(grid, fact[0], axis=1)
    arr = np.repeat(arr, fact[1], axis=0)
    return arr


def mp_worker(args):
    padded, new_game,  coord = args
    t0 = time.time()
    for x, y in coord:
        if (x < 0 or y < 0):
            continue
        part = get_partial_matrix(np.copy(padded), (x, y))
        new_game[x, y] = new_state(part).value
    return new_game


def proc_worker(padded, new_game, coord):
    t0 = time.time()
    for x, y in coord:
        if (x < 0 or y < 0):
            continue
        part = get_partial_matrix(np.copy(padded), (x, y))
        new_game[x, y] = new_state(part).value
    # print(time.time() - t0)
    return new_game


class Game:
    def __init__(self, grid_dimension: tuple[int, int], size_th):
        self.grid = np.zeros(grid_dimension, dtype=int)
        self.dimensions = grid_dimension
        self.history = np.zeros(
            (grid_dimension[0], grid_dimension[1], 50), dtype=int)
        self.size_th = size_th

    def populate_grid(self, grid, partial=True):
        # print(grid)
        if partial == False:
            self.grid = grid
            return
        self.grid[self.grid.shape[0]//2 - grid.shape[0]//2:self.grid.shape[0]//2 + grid.shape[0]//2:,
                  self.grid.shape[1]//2 - grid.shape[1]//2: self.grid.shape[1]//2 + grid.shape[1]//2] = grid

    def step(self):
        padded_grid = np.pad(self.grid, 1)
        new_game = np.zeros(self.dimensions, dtype=int)
        num_neighbors = np.zeros(self.dimensions, dtype=int)
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                part = get_partial_matrix(np.copy(padded_grid), (x, y))
                # num_neighbors[x, y] = count_neighbors(np.copy(part))
                new_game[x, y] = new_state(part).value
                # print(timings)
        # print(num_neighbors)
        return new_game

    def step_optimized(self):
        timings = {}
        padded_grid = np.pad(self.grid, 1)
        new_game = np.zeros(self.dimensions, dtype=int)

        # timings["dis"] = time.time()
        dis = disect_grid(self.grid, self.size_th)
        # timings["dis"] = time.time() - timings["dis"]
        # timings["neigh"] = time.time()
        neighbor_grid = change_neighbors(dis)
        # timings["neigh"] = time.time() - timings["neigh"]
        # print(neighbor_grid)
        # timings["inc"] = time.time()
        neighbor_grid = increase_grid(neighbor_grid, self.grid.shape)
        # timings["inc"] = time.time() - timings["inc"]
        # print(timings)

        coords = np.array(np.where(neighbor_grid == 1)
                          ).ravel("F").reshape(-1, 2)
        num_workers = 16
        # TODO Use Process instead mp.map clones memory -> high load time

        # workers = [multiprocessing.Process(
        #     target=mp_worker, args=(padded_grid, new_game, coords)) for coords in split_coords]

        # for worker in workers:
        #     worker.start()
        if (self.grid.shape[0] > 2**8):
            timings["initpool"] = time.time()
            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                split_coords = np.array_split(coords, num_workers)
                split = [(padded_grid, new_game, coord)
                         for coord in list(split_coords)]
                # The Creation of the processes take longer than the execution of the single Core exec
                res_collection = p.map(mp_worker, split)
                base = res_collection[0]
                for res in res_collection:
                    base = np.where(res == 0, base, res)
                print(timings, sum(timings.values()))
            return base
        for x, y in coords:
            if (x < 0 or y < 0):
                continue
            part = get_partial_matrix(np.copy(padded_grid), (x, y))
            new_game[x, y] = new_state(part).value
        return new_game

    def generation(self):
        # t0 = time.time()
        # g1 = self.step()
        # t0 = time.time() - t0
        t1 = time.time()
        g2 = self.step_optimized()
        t1 = time.time() - t1
        # print(t1)
        # print(g1, g2)
        # if not np.array_equal(g1, g2):
        #     raise Exception("Normal and Optimized outputs are not equal")
        # print(g1, g2)

        # print(self.size_th, t0, t1)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(g2, cmap="gray")
        # axs[0].set_title("Optimized")
        # axs[1].imshow(g1, cmap="gray")
        # axs[1].set_title("Normal")
        # plt.show()

        self.grid = g2
        return self.grid
    # Optimized version


def testmatrix():
    size = 1.9*2**3

    zeros = np.zeros((8, 8), dtype=int)
    zeros[-1, :] = 1
    zeros[-3, 3] = 1
    print(zeros, zeros.shape)
    return zeros
    # return np.arange(size**2).reshape(size, size)


def test_grid(size=2**2):

    return np.random.randint(0, 2, size**2).reshape(size, size)


def find_optimal_size():
    times: dict[int, list] = {}

    # create game
    size = int(2**8)
    grid = test_grid(size//2**2)

    # test multiple sizes
    for size_th in [2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7]:
        times[size_th] = []
        game = Game((size, size), size_th)
        game.populate_grid(grid)
        for i in range(10):
            times[size_th].append(game.generation())
    result = {}
    for key, timings in times.items():
        mean = np.mean(np.array(timings), axis=0)
        result[key] = mean[0]/mean[1]

    print(result)
    return result


def basic():
    size = int(2**9)
    grid = test_grid(2**3)

    game = Game((size, size), find_smallest_divisor((size, size)))
    game.populate_grid(grid)
    game.generation()
    return game.grid


def main():
    # # input()
    print(basic())
    # find_optimal_size()


if __name__ == '__main__':
    main()
