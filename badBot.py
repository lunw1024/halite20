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
    state['me'] = board.current_player_id
    state['playerNum'] = len(board.players)
    state['memory'] = {}
    pass

# Run start of every turn
def update(board):

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
# General random helper functions that are not strictly "process" or in "nav"
# Core strategy

def ship_tasks(): # return updated tasks
    cfg = state['configuration']
    board = state['board']
    me = board.current_player
    
    action = {} #ship -> [value,ship,target]
    tasks = {}
    assign = []

    # Rule based
        # Run, return

    for ship in me.ships:
        # Run 
        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (math.inf,ship,state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue

        # Return
        RETURN_THRESHOLD = 5
        if ship.halite > RETURN_THRESHOLD * state['haliteMean'] + board.cells[ship.position].halite: #TODO Optimize the return threshold
            action[ship] = (ship.halite,ship,state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue
            
        assign.append(ship)

    # Reward based
        # Attack, Mine
    targets = []
    for i in board.cells.values(): # Filter targets
        if i.shipyard != None:
            continue
        if i.halite == 0 and i.ship == None:
            continue
        targets.append(i)
    rewards = np.zeros((len(assign), len(targets)))

    # TODO: Remove nested for loop
    for i,ship in enumerate(assign):
        for j,cell in enumerate(targets):
            rewards[i, j] = naiveReward(ship,cell)

    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True) # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        action[assign[r]] = (naiveReward(assign[r],targets[c]),assign[r],targets[c].position)

    #TODO: Add shipyard attack
    #Process actions
    actions = list(action.values())
    actions.sort(reverse=True,key=lambda x : x[0])
    for act in actions:
        act[1].next_action = a_move(act[1],act[2],state[act[1]]['blocked'])
        # Ship convertion
        sPos = act[1].position 
        if state['closestShipyard'][sPos.x][sPos.y] == sPos and state['board'].cells[sPos].shipyard == None:
            act[1].next_action = ShipAction.CONVERT


    return

def spawn_tasks():
    shipyards = state['board'].current_player.shipyards
    for shipyard in shipyards:
        if state['currentHalite'] > 500 and not state['next'][shipyard.cell.position.x][shipyard.cell.position.y]:
            shipyard.next_action = ShipyardAction.SPAWN   
            state['currentHalite'] -= 500

def convert_tasks():
    # Add convertion tasks
    currentShipyards = state['myShipyards']
    if len(currentShipyards) == 0:
        maxShip = max(state['myShips'],key=lambda ship : ship.halite)
        state['allyShipyard'][maxShip.position.x][maxShip.position.y] = 1
        state['closestShipyard'] = closest_shipyard([maxShip])
    
# General calculations whose values are expected to be used in multiple instances
# Basically calc in botv1.0. 
# Run in update() - see dependency.py

def encode():
    global state
    
    N = state['configuration'].size

    # Halite 
    state['haliteMap'] = np.zeros((N, N))
    for cell in state['cells']:
        state['haliteMap'][cell.position.x][cell.position.y] = cell.halite
    # Ships
    state['shipMap'] = np.zeros((state['playerNum'], N, N))
    for ship in state['ships']:
        state['shipMap'][ship.player_id][ship.position.x][ship.position.y] = 1
    # Shipyards
    state['shipyardMap'] = np.zeros((state['playerNum'], N, N))
    for shipyard in state['shipyards']:
        state['shipyardMap'][shipyard.player_id][shipyard.position.x][shipyard.position.y] = 1
    # Mean Halite 
    state['haliteMean'] = np.mean(state['haliteMap'], axis=None)
    # Friendly units
    state['ally'] = state['shipMap'][state['me']]
    # Friendly shipyards
    state['allyShipyard'] = state['shipyardMap'][state['me']]
    # Enemy units
    state['enemy'] = np.sum(state['shipMap'], axis=0) - state['ally']
    # Enemy shipyards
    state['enemyShipyard'] = np.sum(state['shipyardMap'], axis=0) - state['allyShipyard']
    # Closest shipyard
    state['closestShipyard'] = closest_shipyard(state['myShipyards'])
    # Control map
    state['controlMap'] = control_map(state['ally']-state['enemy'],state['allyShipyard']-state['enemyShipyard'])
    #Enemy ship labeled by halite. If none, infinity
    state['enemyShipHalite'] = np.zeros((N, N))
    state['enemyShipHalite'] += np.Infinity
    for ship in state['ships']:
        if ship.player.id != state['me']:
            state['enemyShipHalite'][ship.position.x][ship.position.y] = ship.halite
    # Avoidance map (Places not to go for each ship)
    for ship in state['myShips']:
        state[ship] = {}
        state[ship]['blocked'] = get_avoidance(ship)

    
def get_avoidance(s):
    threshold = s.halite
    #Enemy units
    temp = np.where(state['enemyShipHalite'] < threshold, 1, 0)
    enemyBlock = np.copy(temp)
    enemyBlock = enemyBlock + np.roll(temp,1,axis=0)
    enemyBlock = enemyBlock + np.roll(temp,-1,axis=0)
    enemyBlock = enemyBlock + np.roll(temp,1,axis=1)
    enemyBlock = enemyBlock + np.roll(temp,-1,axis=1)

    enemyBlock = enemyBlock + state['enemyShipyard'] - state['allyShipyard']*5

    blocked = enemyBlock
    blocked = np.where(blocked>0,1,0)
    return blocked

def closest_shipyard(shipyards):
    N = state['configuration'].size
    res = [[None for y in range(N)]for x in range(N)]
    for x in range(N):
        for y in range(N):
            minimum = math.inf
            for shipyard in shipyards:
                if dist(Point(x,y),shipyard.position) < minimum:
                    minimum = dist(Point(x,y),shipyard.position)
                    res[x][y] = shipyard.position
    return res
    

def control_map(ships,shipyards):
        
        ITERATIONS = 4
        
        res = ships

        #TODO: Use convolutions instead of this hacky method.
        # Convolutions will be more extensible down the line

        for i in range(ITERATIONS):
            temp = np.roll(res,1,axis=0)
            temp += np.roll(res,-1,axis=0)
            temp += np.roll(res,1,axis=1)
            temp += np.roll(res,-1,axis=1)

            res += temp
        
        return res + shipyards

# Direction from point s to point t
def direction_to(s: Point, t: Point) -> ShipAction:
    candidate = directions_to(s, t)
    return random.choice(candidate) if len(candidate) > 0 else None

# Distance from point a to b
def dist(a: Point, b: Point) -> int:
    N = state['configuration'].size
    return min(abs(a.x - b.x), N - abs(a.x - b.x)) + min(abs(a.y - b.y), N - abs(a.y - b.y))

# Returns list of possible directions
def directions_to(s: Point, t: Point) -> ShipAction:
    N = state['configuration'].size
    candidates = [] # [N/S, E/W]
    if s.x-t.x != 0:
        candidates.append(ShipAction.WEST if (s.x-t.x) % N < (t.x-s.x) % N else ShipAction.EAST)
    if s.y-t.y != 0:
        candidates.append(ShipAction.SOUTH if (s.y-t.y) % N < (t.y-s.y) % N else ShipAction.NORTH)
    return candidates

# Deserialize an integer which represents a point
def unpack(n):
    N = state['configuration'].size
    return Point(n // N, n % N)

# A default direction to target
def direction_to(s: Point, t: Point) -> ShipAction:
    candidate = directions_to(s, t)
    return random.choice(candidate) if len(candidate) > 0 else None


# Returns the "next" point of a ship at point s with shipAction d
def dry_move(s: Point, d: ShipAction) -> Point:
    N = state['configuration'].size
    if d == ShipAction.NORTH:
        return s + Point(0, 1) % N
    elif d == ShipAction.SOUTH:
        return s + Point(0, -1) % N
    elif d == ShipAction.EAST:
        return s + Point(1, 0) % N
    elif d == ShipAction.WEST:
        return s + Point(-1, 0) % N
    else:
        return s
    
# Returns list of len 4 of adjacent points to a point
def get_adjacent(point):
    N = state['configuration'].size
    res = []
    for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
        res.append(point.translate(Point(offX,offY),N))
    return res

# A* Movement from ship s to point t
# See https://en.wikipedia.org/wiki/A*_search_algorithm
def a_move(s : Ship, t : Point, inBlocked):

    nextMap = state['next']
    sPos = s.position
    blocked = inBlocked + nextMap
    blocked = np.where(blocked>0,1,0)

    #Stay still
    if sPos == t:
        #Someone with higher priority needs position, must move. Or being attacked.
        if blocked[t.x][t.y]:
            for processPoint in get_adjacent(sPos):
                if not blocked[processPoint.x][processPoint.y]:
                    nextMap[processPoint.x][processPoint.y] = 1
                    return direction_to(sPos,processPoint)
            nextMap[sPos.x][sPos.y] = 1
            return None
        else:
            nextMap[sPos.x][sPos.y] = 1
            return None

    #A*
    pred = {}
    calcDist = {}
    pq = PriorityQueue()
    pqMap = {}

    pqMap[dist(sPos,t)] = [sPos]
    pq.put(dist(sPos,t))
    pred[sPos] = sPos
    calcDist[sPos] = dist(sPos,t)

        # Main

    while not pq.empty():
        if t in calcDist:
            break
        currentPoint = pqMap.get(pq.get()).pop()
        for processPoint in get_adjacent(currentPoint):
            if blocked[processPoint.x][processPoint.y] or processPoint in calcDist: 
                continue
            calcDist[processPoint] = calcDist[currentPoint] + 1
            priority =  calcDist[processPoint] + dist(processPoint,t)
            pqMap[priority] = pqMap.get(priority,[])
            pqMap[priority].append(processPoint)
            pq.put(priority)
            pred[processPoint] = currentPoint

    if not t in pred:
        #Random move
        for processPoint in get_adjacent(sPos):
            if not blocked[processPoint.x][processPoint.y]:
                nextMap[processPoint.x][processPoint.y] = 1
                return direction_to(sPos,processPoint)
        nextMap[sPos.x][sPos.y] = 1
        return None

        # Path reconstruction
    while pred[t] != sPos:
        t = pred[t]

    desired = direction_to(sPos,t)
    nextMap[t.x][t.y] = 1
    
    return desired
# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell
# This should take the form of a neural network
def getReward(ship,cell):
    global state
    return 1

def naiveReward(ship,cell):
    # For testing purposes
    if cell.ship != None and cell.ship.player_id != state['me'] and cell.ship.halite < ship.halite:
        return -100
    else:
        res = cell.halite - dist(ship.position,cell.position) * 10
        if cell.ship == ship:
            res += 50
        return res
# The final function

@board_agent
def agent(board):

    print("Turn =",board.step+1)
    # Init
    if board.step == 0:
        init(board)

    # Update
    update(board)
    
    # Convert
    convert_tasks()
    
    # Ship
    ship_tasks()
    
    # Spawn
    spawn_tasks()