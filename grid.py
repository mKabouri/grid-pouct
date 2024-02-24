import pygame
import random
import numpy as np
from typing import List, Tuple

import config

ACTIONS = {
    "left": 0,
    "top": 1,
    "right": 2,
    "bottom": 3
}

class Cell(object):
    def __init__(self, x: int, y: int, tile_size: int, state: int, is_goal: bool):
        # The number of the state represent its state
        self.state = state
        self.is_goal = is_goal
        self.x = x
        self.y = y
        self.tile_size = tile_size

    @property
    def is_goal_cell(self) -> bool:
        return self.is_goal

    @property
    def get_position(self) -> Tuple:
        return self.x, self.y

    def draw_cell(self, screen: pygame.Surface):
        if self.is_goal:
            color = config.GOAL_CELL_COLOR
        else:
            color = random.choice(config.POSSIBLE_COLORS)
        pos_x, pos_y = self.x*self.tile_size, self.y*self.tile_size
        pygame.draw.rect(
            screen,
            color,
            (pos_x, pos_y, self.tile_size, self.tile_size),
        )
        wall_thickness = 2
        lines = [
            ((pos_x, pos_y), (pos_x+self.tile_size, pos_y)),
            ((pos_x+self.tile_size, pos_y), (pos_x+self.tile_size, pos_y+self.tile_size)),
            ((pos_x+self.tile_size, pos_y+self.tile_size), (pos_x, pos_y+self.tile_size)),
            ((pos_x, pos_y+self.tile_size), (pos_x, pos_y))
        ]
        for line in lines:
            pygame.draw.line(screen, config.BLACK_COLOR, line[0], line[1], wall_thickness)


class Grid(object):
    def __init__(self, width: int, height: int, tile_size: int, nb_states: int) -> None:
        pygame.init()
        self.height = height
        self.width = width
        self.tile_size = tile_size
        self.nb_states = nb_states
        self._init_pygame()
        self.goal_state = np.random.choice(self.nb_states)

    def _init_pygame(self):
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(("POMDP GRID"))

    def reset(self):
        pass

    def step(self, action: str):
        """
        You should remember that we are working with pomdp
        so you have to return an observation
        """
        pass

    def _draw_cells(self) -> List[Cell]:
        cells = []
        for y in range(self.height//self.tile_size):
            for x in range(self.width//self.tile_size):
                if y*self.height//self.tile_size + x == self.goal_state:
                    cell = Cell(x, y, self.tile_size, y*self.height//self.tile_size + x, True)
                else:
                    cell = Cell(x, y, self.tile_size, y*self.height//self.tile_size + x, False)
                cell.draw_cell(self.screen)
                cells.append(cell)
        return cells

    def _draw_agent(self, cells: List[Cell]):
        """
        But if you draw the agent that means that you know where is it ?
        Todo: Add probabilities here!
        """
        pick_cell = random.choice(cells)
        while pick_cell.is_goal_cell:
            pick_cell = random.choice(cells)        
        pos_x, pos_y = pick_cell.get_position
        pos_x, pos_y = pos_x*self.tile_size, pos_y*self.tile_size
        pygame.draw.circle(
            self.screen,
            config.AGENT_COLOR,
            (pos_x + self.tile_size//2, pos_y + self.tile_size//2),
            self.tile_size//4
        )

    def draw_grid(self, fps: int) -> None:
        clock = pygame.time.Clock()
        cells = self._draw_cells()
        self._draw_agent(cells)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting")
                    running = False
                # Quit if we click on "q"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("Exiting")
                        running = False
            pygame.display.flip()
            clock.tick(fps)

        pygame.quit()

if __name__ == '__main__':
    grid_env = Grid(config.WIDTH, config.HEIGHT, config.TILE_SIZE, config.NB_STATES)
    grid_env.draw_grid(50)
