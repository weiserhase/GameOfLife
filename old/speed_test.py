import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np


def grid2dict(grid):
    coords = {}
    for i in range(grid.shape[0]*grid.shape[1]):
        coords[i] = random.randint(
            0, 2)
    # print(len(coords))
    return set(coords.keys()), coords


def init(size):
    npa = np.random.randint(0, 2, size**2).reshape(size, size)
    s, dc = grid2dict(npa)
    return npa, s, dc


def test_np(matrix, key):
    matrix[key//matrix.shape[0], key %
           matrix.shape[0]] = key + random.randint(0, 10000)


def test_set(set, key):
    set.add(key + random.randint(0, 10000))


def test_dict(dic, key):
    dic[key] = key + random.randint(0, 10000)


def test(f, x, test_size, size):
    t0 = time.time()
    ix = np.random.randint(0, size, test_size)
    for i in range(test_size):
        idx = ix[i]
        f(x, idx)
    tnp = time.time() - t0

    return tnp


def measure(size, test_size):
    a, s, d = init(size)
    # print(a, d)

    tnp = test(test_np, a, test_size, size)
    tdc = test(test_dict, d, test_size, size)
    ts = test(test_set, s, test_size, size)
    print(f"Size: {size} test_size: {test_size} Np: {tnp} DC: {tdc} S: {ts}")
    return tnp, tdc, ts


def plot_test(max_size, test_size):
    dc = []
    npc = []
    sc = []
    y = []
    for i in range(1, max_size, 25):
        n, d, s = measure(i, test_size)
        npc.append(n)
        dc.append(d)
        sc.append(s)
        y.append(i)

    def avg(list):
        return sum(list)/len(list)
    print(avg(npc), avg(dc), avg(sc))
    plt.plot(y, npc)
    plt.plot(y, dc)
    plt.plot(y, sc)
    plt.show()


def main():
    # print(measure(2**4, 10**7))
    plot_test(2000, 10**5)


if __name__ == '__main__':
    main()
