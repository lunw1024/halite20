# Contains all dependencies used in bot
# First file loaded

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import math, random
import numpy as np
import scipy.optimize
import scipy.ndimage
from queue import PriorityQueue

# Global constants

    # All game state goes here - everything, even mundane
state = {}

    # Contains all weights to be initialized
    #TODO: Train!

weights = [
    # shipyard_reward weights - 0
    np.array([1,1,-1,0.5,0.2])
] 

def setWeight(v):
    global weights
    weights = v

# Init function - called at the start of each game
def init(board):
    global state
    state['configuration'] = board.configuration
    state['me'] = board.current_player_id
    state['playerNum'] = len(board.players)
    state['memory'] = {}
    pass

# Run start of every turn
def update(board):
    global action
    action = {}
    state['currentHalite'] = board.current_player.halite
    state['next'] = np.zeros((board.configuration.size,board.configuration.size))
    state['board'] = board
    state['memory'][board.step] = board
    state['cells'] = board.cells.values()
    state['ships'] = board.ships.values()
    state['myShips'] = board.current_player.ships
    state['shipyards'] = board.shipyards.values()
    state['myShipyards'] = board.current_player.shipyards

    # Calc processes
    encode()