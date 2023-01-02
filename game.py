import numpy as np
import pygame as pg

from buttons import Button
from main import *

pg.font.init()
gui_font = pg.font.Font(None, 30)


def blit_text(surface, text, pos, font=gui_font, color=pg.Color('white')):
    # 2D array where each row is a list of words.
    words = [word.split(' ') for word in text.splitlines()]
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.


class MenuConfig:
    def __init__(self, buttons: list[Button], background="black"):
        self.buttons = buttons
        self.background = background


class GameCore:
    def __init__(self, game: Game, width: int = 900, height: int = 900, fps: int = 60, animation_rate: int = 1) -> None:
        self.game = game
        num_cells = game.grid.shape
        self.x_cells, self.y_cells = num_cells
        self.x_cell_width, self.y_cell_width = width//num_cells[0], height//num_cells[1]
        self.width = width
        self.height = height
        self.res = width, height
        self.h_width, self.h_height = width//2, height//2
        self.fps = fps
        self.animation_rate = animation_rate
        self.init_screen()

        self.menu_config = MenuConfig([
            Button("Draw", 150, 50, (550, 400), 1,
                   self.screen, self.change_scene),
            Button("Random", 150, 50, (250, 400), 1,
                   self.screen, self.change_scene),
            Button("Start", 150, 50, (550, 500), 1,
                   self.screen, self.change_scene),
            Button("Step", 150, 50, (250, 500), 1,
                   self.screen, self.change_scene)
        ], background="gray")

        self.scene = "Menu"
        self.callbacks = {
            "Menu": self.draw_menu,
            "Game": self.draw_game,
            "Draw": self.draw_draw
        }
        self.step = True
        self.needs_step = False

    def random_grid(self):
        self.game.populate_grid(test_grid(self.x_cells))

    def change_scene(self, text):
        if text == "Draw":
            self.game.populate_grid(np.zeros(self.game.grid.shape), False)
            self.scene = text
        elif text == "Random":
            self.random_grid()
            self.scene = "Menu"
        elif text == "Game" or text == "Start":
            self.scene = "Game"
        elif text == "Step":
            self.needs_step = True
            self.scene = "Game"
        else:
            self.scene = "Menu"

    def init_screen(self):
        pg.init()
        self.screen = pg.display.set_mode(self.res, pg.RESIZABLE)
        self.clock = pg.time.Clock()

    def draw_grid(self):
        self.screen.fill(pg.Color("black"))
        [pg.draw.line(self.screen, pg.Color("dimgray"), (x, 0), (x, self.height))
         for x in range(0, self.width, self.x_cell_width)]
        [pg.draw.line(self.screen, pg.Color("dimgray"), (0, y), (self.width, y))
         for y in range(0, self.height, self.y_cell_width)]

    def fill_grid(self, grid):
        self.draw_grid()

        for x in range(self.x_cells):
            for y in range(1, self.y_cells):
                if grid[x][y]:
                    pg.draw.rect(self.screen, pg.Color("forestgreen"), (x*self.x_cell_width+2,
                                 y*self.y_cell_width+2, self.x_cell_width - 2, self.y_cell_width - 2))

    def draw_game(self):

        if(self.step == True or self.needs_step == False):
            self.game.generation()
            self.step = False
        grid = self.game.grid
        self.fill_grid(grid)

    def draw_draw(self):
        self.fill_grid(self.game.grid)

    def draw_menu(self):
        self.draw_grid()
        # self.screen.fill(pg.Color(self.menu_config.background))
        for button in self.menu_config.buttons:
            button.draw()
        description = """
            This is a implementation for the Game of life.
            You are finding yourself in the menu right now.
            The first row of buttons are buttons for initializing a grid \n\n
            Press SPACE to return to menu\n\n
            In the Drawing mode you can change the cellstate by clicking on the cell (leftclick). 
            You can go back to the Menu by pressing ENTER
            In the Step mode you can evolve the next generation by hitting ENTER\n
        """
        blit_text(self.screen, description, (50, 550))
        # surface = gui_font.render(description, False, (255, 255, 255))
        # self.screen.blit(surface, (550, 50))

    def event_handler(self, event):

        if event.type == pg.QUIT:
            exit()
        if event.type == pg.MOUSEBUTTONDOWN and self.scene == "Draw":
            pos = pg.mouse.get_pos()
            grid = np.copy(self.game.grid)
            # Calculate indices of clicked cell
            x = pos[0] // self.x_cell_width
            y = pos[1] // self.y_cell_width
            if (grid[x, y] == 1):
                grid[x, y] = 0
            else:
                grid[x, y] = 1
            self.game.populate_grid(grid, partial=False)
        if event.type == 768 and self.scene == "Draw":
            print("enter")
            self.change_scene("Menu")
        if event.type == 768 and self.scene == "Game":
            self.step = True
        if event.type == 771:
            self.scene = "Menu"

    def run(self):
        while True:
            for evt in pg.event.get():
                self.event_handler(evt)

                # Draw scene
            self.callbacks[self.scene]()

            # Handle Events
            pg.display.set_caption(self.scene)
            # print(str(self.clock.get_fps()))
            pg.display.flip()
            if(self.scene == "Game"):
                self.clock.tick(self.animation_rate)
            self.clock.tick(self.fps)


def menu_debug():
    # Die Größe des Felds
    size = int(2**3)
    # Hier ein eigenes grid einfügebn
    game = Game((size, size), find_smallest_divisor((size, size)))
    # game.populate_grid(grid)

    screen = GameCore(game, fps=60)
    screen.run()


def main():
    # Die Größe des Felds
    size = int(2**5)
    # Hier ein eigenes grid einfügebn
    grid = test_grid(size // 2**2)

    game = Game((size, size), find_smallest_divisor((size, size)))
    game.populate_grid(grid)

    # set framerate her at fps
    screen = GameCore(game, fps=60, animation_rate=5)
    screen.run()


if __name__ == "__main__":
    main()
