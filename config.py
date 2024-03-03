## Grid parameters:
TILE_SIZE = 150
MULT_FACTOR = 3
WIDTH = TILE_SIZE*MULT_FACTOR
HEIGHT = TILE_SIZE*MULT_FACTOR

NB_STATES = (HEIGHT//TILE_SIZE)*(WIDTH//TILE_SIZE)

GOAL_CELL_COLOR = (0, 255, 0) # Index 3

POSSIBLE_COLORS = [
    (255, 0, 0), # Index 0
    (120, 0, 120), # Index 1
    (200, 50, 120) # Index 2
]

BLACK_COLOR = (0, 0, 0)

AGENT_COLOR = (255, 255, 0) # YELLOW
ACTIONS = {
    "left": 0,
    "top": 1,
    "right": 2,
    "bottom": 3
}
