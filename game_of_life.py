from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Cellstate(Enum):
    DEAD = 0
    ALIVE = 1


def new_cellstate(num_neighbors: int, state: Cellstate):
    if num_neighbors == 2:
        return state
    if num_neighbors == 3:
        return Cellstate.ALIVE
    return Cellstate.DEAD


class Game(ABC):
    """The game class"""
    @abstractmethod
    def step(self):
        """Perform one timestep"""
    @abstractmethod
    def grid(self):
        """ Return the grid as a nparray """
