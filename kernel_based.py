import time

import numpy as np
from numpy.fft import fft2, ifft2


def fft_convolve2d(x, y):
    """
    2D convolution, using FFT
    """
    fourier_x = fft2(x)
    fourier_y = fft2(np.flipud(np.fliplr(y)))
    m, n = fourier_x.shape
    cc = np.real(ifft2(fourier_x*fourier_y))
    cc = np.roll(cc, - int(m / 2) + 1, axis=0)
    cc = np.roll(cc, - int(n / 2) + 1, axis=1)
    return cc


def new_state(num_neighbors) -> int:
    if num_neighbors == 3:
        return 1
    if num_neighbors < 2:
        return 0
    if num_neighbors > 3:
        return 0
    return -1


class FFTGame:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.dimensions = grid.shape
        self.kernel = self.generate_kernel()

    def generate_kernel(self):
        full_kernel = np.zeros(self.grid.shape, dtype=int)
        full_kernel[:3, :3] = np.ones((3, 3), dtype=int)
        full_kernel[1, 1] = 0
        return full_kernel

    def populate_grid(self, grid: np.ndarray):
        self.grid = grid

    def step(self):
        conv = fft_convolve2d(self.grid, self.kernel).round()
        new_grid = np.copy(self.grid)

        new_grid[conv == 3] = 1
        new_grid[conv > 3] = 0
        new_grid[conv < 3] = 0

        # new_grid[np.where((conv == 2) and (self.grid == 1))] = 1
        # new_grid[np.where(conv and self.grid == 1)] = 1

        # new_grid[np.where((conv == 3 and self.grid == 0))] = 1
        self.grid = conv
        return conv


def main():
    size = 2**10
    grid_size = int(size)
    grid = np.random.randint(0, 2, grid_size**2
                             ).reshape(grid_size, grid_size)
    game = FFTGame(grid)
    print(game.generate_kernel())
    timings = []
    for num in range(10):
        arr = np.random.randint(0, 2, size**2, dtype=int).reshape(size, size)
        game = FFTGame(arr)

        timings.append(time.time())
        for i in range(100):
            game.step()
        timings[-1] = time.time() - timings[-1]
        print(f"time elapsed: {timings[-1]}")
    print(timings, sum(timings)/len(timings))


if __name__ == '__main__':
    main()
