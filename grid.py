import pygame
import random
import numpy as np
from typing import List, Tuple, Dict

import config

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
    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int,
        nb_states: int,
        possible_actions: Dict=config.ACTIONS,
    ) -> None:
        self.height = height
        self.width = width
        self._init_pygame()

        self.tile_size = tile_size
        self.nb_states = nb_states

        self.goal_state = np.random.choice(self.nb_states)
        self.possible_actions = possible_actions

        self.cells = self.get_cells()
    
    def get_cells(self) -> List[Cell]:
        cells = []
        for y in range(self.height//self.tile_size):
            for x in range(self.width//self.tile_size):
                if y*self.height//self.tile_size + x == self.goal_state:
                    cell = Cell(x, y, self.tile_size, y*self.height//self.tile_size + x, True)
                else:
                    cell = Cell(x, y, self.tile_size, y*self.height//self.tile_size + x, False)
                cells.append(cell)
        return cells

    @property
    def get_possible_actions(self):
        return self.possible_actions

    @property    
    def get_number_states(self):
        return self.nb_states

    def _init_pygame(self):
        pygame.init()
        # In the + 100 you have to display agent belief where he is vs really where he is 
        self.screen = pygame.display.set_mode((self.width, self.height+100))
        self.screen.fill((255, 255, 255))
        pygame.display.set_caption(("POMDP GRID"))

    def reset(self):
        pass

    def step(self, action: str):
        """
        * You should remember that we are working with pomdp
        so you have to return an observation
        * Rewards are defined here (-1 always) for each time step
        """        
        movement = {
            "left": (-1, 0),
            "top": (0, -1),
            "right": (1, 0),
            "bottom": (0, 1)
        }

        dx, dy = movement.get(action, (0, 0))
        new_x = self.agent_x + dx*self.tile_size
        new_y = self.agent_y + dy*self.tile_size

        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return -1

        self.agent_x = new_x
        self.agent_y = new_y
        return -1

    def _draw_cells(self) -> None:
        for cell in self.cells:
            cell.draw_cell(self.screen)

    def _init_agent(self):
        """
        But if you draw the agent that means that you know where is it ?
        Todo: Add probabilities here!
        """
        pick_cell = random.choice(self.cells)
        while pick_cell.is_goal_cell:
            pick_cell = random.choice(self.cells)        
        self.agent_x, self.agent_y = pick_cell.get_position
        pos_x, pos_y = self.agent_x*self.tile_size, self.agent_y*self.tile_size
        pygame.draw.circle(
            self.screen,
            config.AGENT_COLOR,
            (pos_x + self.tile_size//2, pos_y + self.tile_size//2),
            self.tile_size//4
        )

    def draw_grid(self, fps: int) -> None:
        clock = pygame.time.Clock()
        self._draw_cells()
        self._init_agent()
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

def make_env() -> Grid:
    return Grid(config.WIDTH, config.HEIGHT,
                config.TILE_SIZE, config.NB_STATES)

if __name__ == '__main__':
    grid_env = make_env()
    grid_env.draw_grid(50)
