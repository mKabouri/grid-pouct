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
        # Color is our only observation
        self.color = None

    @property
    def is_goal_cell(self) -> bool:
        return self.is_goal

    @property
    def get_position(self) -> Tuple:
        return self.x, self.y
    
    @property
    def get_color(self) -> Tuple:
        return self.color

    def draw_cell(self, screen: pygame.Surface):
        if self.is_goal:
            self.color = config.GOAL_CELL_COLOR
        else:
            self.color = random.choice(config.POSSIBLE_COLORS)
        pos_x, pos_y = self.x*self.tile_size, self.y*self.tile_size
        pygame.draw.rect(
            screen,
            self.color,
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

        font = pygame.font.Font('freesansbold.ttf', 32)
        text_surface = font.render(str(self.state), True, config.BLACK_COLOR)
        text_rect = text_surface.get_rect(center=(pos_x+self.tile_size//2, pos_y+self.tile_size//2))
        screen.blit(text_surface, text_rect)


class Grid(object):
    """
    """
    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int,
        render: bool=True,
        possible_actions: Dict=config.ACTIONS,
    ) -> None:
        self.height = height
        self.width = width
        self._init_pygame()

        self.tile_size = tile_size

        self.goal_state = np.random.choice(self.get_number_states)
        self.possible_actions = possible_actions

        self.cells = self.get_cells()

        # Define transition probabilities ??

        self.render = render
        if self.render:
            self.draw_grid()
    
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

    def coord2number(self, coord: Tuple):
        return coord[0]*(self.height//self.tile_size) + coord[1]

    def number2coord(self, number: int):
        cols = self.width//self.tile_size
        x = number//cols
        y = number%cols
        return x, y

    @property
    def get_current_observation(self):
        current_cell = self._get_cell_in(self.agent_x, self.agent_y)
        return current_cell.get_color

    @property
    def get_possible_actions(self):
        return self.possible_actions

    @property    
    def get_number_states(self):
        return (self.height//self.tile_size)*(self.width//self.tile_size)

    def _init_pygame(self):
        pygame.init()
        # In the + 100 you have to display agent belief where he is vs really where he is 
        self.screen = pygame.display.set_mode((self.width, self.height+100))
        self.screen.fill((255, 255, 255))
        pygame.display.set_caption(("POMDP GRID"))

    def reset(self):
        """
        Returns first observation
        """
        self.goal_state = np.random.choice(self.get_number_states)
        self._init_pygame()
        self._init_agent()
        self.cells = self.get_cells()
        if self.render:
            self.draw_grid()
        return self.get_current_observation

    def step(self, action: str):
        """
        Returns:
        * Reward
        * The new observation (Color of the current cell ?)
        * Done: bool
        """
        movement = {
            "left": (-1, 0),
            "top": (0, -1),
            "right": (1, 0),
            "bottom": (0, 1)
        }

        dx, dy = movement.get(action, (0, 0))
        new_x = (self.agent_x + dx)*self.tile_size
        new_y = (self.agent_y + dy)*self.tile_size

        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:                
            current_cell = self._get_cell_in(self.agent_x, self.agent_y)
            return -1, current_cell.get_color, False

        self.update_agent_position(new_x, new_y)
        
        current_cell = self._get_cell_in(self.agent_x, self.agent_y)
        if current_cell.is_goal_cell:
            return 10, current_cell.get_color, True

        return -1, current_cell.get_color, False

    def update_agent_position(self, new_x, new_y) -> None:
        self.agent_x = new_x//self.tile_size
        self.agent_y = new_y//self.tile_size

    def _draw_cells(self) -> None:
        for cell in self.cells:
            cell.draw_cell(self.screen)

    def _get_cell_in(self, x: int, y: int) -> Cell:
        if x*self.tile_size < 0 or x*self.tile_size >= self.width or y*self.tile_size < 0 or y*self.tile_size >= self.height:
            raise ValueError("Index out of bounds.")
        for cell in self.cells:
            if cell.get_position == (x, y):
                return cell

    def _init_agent(self):
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

    @property
    def _get_agent_position(self):
        return self.agent_x, self.agent_y

    def draw_grid(self) -> None:
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
            clock.tick(30)

        pygame.quit()

def make_env() -> Grid:
    return Grid(config.WIDTH, config.HEIGHT,
                config.TILE_SIZE)

if __name__ == '__main__':
    grid_env = make_env()
    print(grid_env.coord2number((0, 1)))
    print(grid_env.number2coord(1))
    