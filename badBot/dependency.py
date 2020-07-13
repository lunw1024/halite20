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
state = []

    # Contains all weights to be initialized - everything, no magic numbers of HUMAN_SET
    # Optimization through bayesian, genetic, annealing etc
    # A list of NP vectors - each vector represents one "layer" of weights in a network
    # AKA - a custom, simple, neural net. TensorFlow is overkill for what we need, as there is no backpropagation
    # TODO: If strategy involves supervised learning, change weights to be compatible with TF.

weights = [] 

def setWeight(v):
    global weights
    weights = v

# Init function - called at the start of each game
def init(board):
    global state
    state['configuration'] = board.configuration
    state['memory'] = {}
    pass

# Run start of every turn
def update(board):
    state['board'] = board
    state['memory'][board.step] = board
    state['ships'] = board.ships.values()
    state['shipyards'] = board.shipyards.values()