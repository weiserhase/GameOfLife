import numpy as np


def test_grid(sizex, sizey):
    return np.random.randint(0, 2, sizex*sizey).reshape(sizex, sizey)


def find_divison(sizex, sizey) -> tuple[int, int]:
    def division(x):
        uneven = []
        for val in range(1, x):
            if(x % val == 0):
                if(x/val % 2 == 0):
                    uneven.append(val)
                    continue
                return val
        if uneven == []:
            raise Exception("No division")
        return uneven[0]
    result = (division(sizex), division(sizey))
    return result


def increase_grid(matrix: np.ndarray, size: tuple[int, int]):
    fact = size[0]/matrix.shape[0], size[1]/matrix.shape[1]
    arr = np.repeat(matrix, fact[0], axis=1)
    arr = np.repeat(arr, fact[1], axis=0)
    return arr


def disect_grid(matrix: np.ndarray) -> np.ndarray:
    result = matrix

    return result


def main():
    grid = test_grid(3, 3)
    # print(find_divison(3, 6))
    print(grid)
    print(increase_grid(grid, (5, 6)))


if __name__ == '__main__':
    main()
