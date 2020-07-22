weights='''1 -1 0.5 0.5
0.01
1
1
1 -3'''
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
    # Bot training weights
        # 0 - shipyard reward
        # 1 - mine reward
temp = []
weights = weights.split('\n')
for line in weights:
    temp.append(np.array(list(map(float,line.split()))))
weights = temp

# Init function - called at the start of each game
def init(board):
    global state
    np.set_printoptions(precision=3)
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
# General random helper functions that are not strictly "process" or in "nav"

# Map from 0 to 1
def normalize(v):
    norm = np.linalg.norm(v,np.inf)
    if norm == 0: 
       return v
    return v / norm

def closest_ship(t):
    res = None
    for ship in state['myShips']:
        if res == None:
            res = ship
        elif dist(t,res.position) > dist(t,ship.position):
            res = ship
    return res

def halite_per_turn(deposit, shipTime, returnTime):
    travelTime = shipTime + returnTime
    actualDeposit = min(500,deposit * 1.02 ** shipTime)
    maximum = 0
    for turns in range(1,10):
        mined = (1 - .75**turns) * actualDeposit
        perTurn = mined / (turns+travelTime)
        maximum = perTurn if perTurn > maximum else maximum
    return maximum

# Core strategy

action = {}  # ship -> (value,ship,target)


def ship_tasks():  # update action
    global action
    cfg = state['configuration']
    board = state['board']
    me = board.current_player
    tasks = {}
    shipsToAssign = []

    # Rule based: Flee, Return

    for ship in me.ships:
        # Flee
        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (math.inf, ship, state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue # continue its current action

        # End-game return
        if board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite > 0:
            action[ship] = (ship.halite, ship, state['closestShipyard'][ship.position.x][ship.position.y])
        if ship in action:
            continue

        # TODO: Force attack

        shipsToAssign.append(ship)

    # Reward based: Attack, Mine
    targets = []
    for i in board.cells.values():  # Filter targets
        if i.shipyard != None and i.shipyard.player_id == state['me']:
            for j in range(min(8,len(state['myShips']))):
                targets.append(i)
            continue
        if i.halite == 0  and i.ship == None:
            continue
        targets.append(i)
    rewards = np.zeros((len(shipsToAssign), len(targets)))

    for i, ship in enumerate(shipsToAssign):
        for j, cell in enumerate(targets):
            rewards[i, j] = get_reward(ship, cell)

    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True)  # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        action[shipsToAssign[r]] = (rewards[r][c], shipsToAssign[r], targets[c].position)

    # Process actions
    actions = list(action.values())
    actions.sort(reverse=True, key=lambda x: x[0])
    for act in actions:
        process_action(act)

def process_action(act):
    global action
    if action[act[1]] == True:
        return
    action[act[1]] = True
    # Processing
    act[1].next_action = a_move(act[1], act[2], state[act[1]]['blocked'])
    # Ship convertion
    sPos = act[1].position
    if state['closestShipyard'][sPos.x][sPos.y] == sPos and state['board'].cells[sPos].shipyard == None:
        act[1].next_action = ShipAction.CONVERT
        
    return act[1].next_action

def spawn_tasks():
    shipyards = state['board'].current_player.shipyards
    shipyards.sort(reverse=True, key=lambda shipyard: state['haliteSpread'][shipyard.position.x][shipyard.position.y])
    for shipyard in shipyards:
        if state['currentHalite'] > 500 and not state['next'][shipyard.cell.position.x][shipyard.cell.position.y]:
            if state['shipValue'] > 500:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif shipyard.cell.ship != None and not shipyard.cell.ship.player_id == state['me']:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShips']) <= 2:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500


def convert_tasks():
    global action

    # Add convertion tasks

    rewardMap = shipyard_reward_map()  # Best area to build a shipyard
    currentShipyards = state['myShipyards']  # Shipyards "existing"
    targetShipyards = currentShipyards[:]

    t = np.where(rewardMap == np.amax(rewardMap))
    tx, ty = list(zip(t[0], t[1]))[0]

    # Calculate the reward for each cell

    if len(currentShipyards) == 0:
        # Grab the closest ship to the target and build.
        closest = closest_ship(Point(tx, ty))
        action[closest] = (math.inf, closest, Point(tx, ty))
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500
    elif len(state['myShips']) >= len(currentShipyards) * 5 + 6 and len(state['myShipyards']) < 4:
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500

    state['closestShipyard'] = closest_shipyard(targetShipyards)

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
    # Halite Spread
    state['haliteSpread'] = np.copy(state['haliteMap'])
    for i in range(1,5):
        state['haliteSpread'] += np.roll(state['haliteMap'],i,axis=0) * 0.5**i
        state['haliteSpread'] += np.roll(state['haliteMap'],-i,axis=0) * 0.5**i
    temp = state['haliteSpread'].copy()
    for i in range(1,5):
        state['haliteSpread'] += np.roll(temp,i,axis=1) * 0.5**i
        state['haliteSpread'] += np.roll(temp,-i,axis=1) *  0.5**i
    # Ships
    state['shipMap'] = np.zeros((state['playerNum'], N, N))
    for ship in state['ships']:
        state['shipMap'][ship.player_id][ship.position.x][ship.position.y] = 1
    # Shipyards
    state['shipyardMap'] = np.zeros((state['playerNum'], N, N))
    for shipyard in state['shipyards']:
        state['shipyardMap'][shipyard.player_id][shipyard.position.x][shipyard.position.y] = 1
    # Total Halite
    state['haliteTotal'] = np.sum(state['haliteMap'])
    # Mean Halite 
    state['haliteMean'] = state['haliteTotal'] / (N**2)
    # Estimated "value" of a ship
    #totalShips = len(state['ships'])
    #state['shipValue'] = state['haliteTotal'] / state
    state['shipValue'] = ship_value()
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
        
        ITERATIONS = 3

        res = np.copy(ships)

        for i in range(1,ITERATIONS+1):
            res += np.roll(ships,i,axis=0) * 0.5**i
            res += np.roll(ships,-i,axis=0) * 0.5**i
        temp = res.copy()
        for i in range(1,ITERATIONS+1):
            res += np.roll(temp,i,axis=1) * 0.5**i
            res += np.roll(temp,-i,axis=1) * 0.5**i
        
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
def unpack(n) -> Point:
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
    # Check if we are trying to attack
    if state['board'].cells[t].ship != None:
        target = state['board'].cells[t].ship
        if target.player_id != state['me'] and target.halite == s.halite:
            blocked[t.x][t.y] -= 1
    elif state['board'].cells[t].shipyard != None and state['board'].cells[t].shipyard.player_id != state['me']:
        blocked[t.x][t.y] -= 1
    # Don't ram stuff thats not the target. Unless we have an excess of ships.
    if len(state['myShips']) < 30:
        blocked += np.where(state['enemyShipHalite'] == s.halite, 1, 0)

    blocked = np.where(blocked>0,1,0)

    #Stay still
    if sPos == t or nextMap[t.x][t.y]:
        #Someone with higher priority needs position, must move. Or being attacked.
        if blocked[t.x][t.y]:
            for processPoint in get_adjacent(sPos):
                if not blocked[processPoint.x][processPoint.y]:
                    nextMap[processPoint.x][processPoint.y] = 1
                    return direction_to(sPos,processPoint)
            
            return micro_run(s)
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
        if blocked[sPos.x][sPos.y]:
            return micro_run(s)
        return None

        # Path reconstruction
    while pred[t] != sPos:
        t = pred[t]

    desired = direction_to(sPos,t)
    # Reduce collisions
    if state['board'].cells[t].ship != None and state['board'].cells[t].ship.player_id == state['me']:
        target = state['board'].cells[t].ship
        if action[target] != True:
            nextMap[t.x][t.y] = 1
            result = process_action(action[target])
            # Going there will kill it
            if result == None:
                desired = a_move(s,t,inBlocked)
                nextMap[t.x][t.y] = 0
                return desired

    nextMap[t.x][t.y] = 1
    return desired

# Ship might die, RUN!
def micro_run(s):
    sPos = s.position
    print("Might die?",sPos)
    nextMap = state['next']

    if state[s]['blocked'][sPos.x][sPos.y]:
        print("by enemy")
        score = [0,0,0,0]
        for i,pos in enumerate(get_adjacent(sPos)):
            if nextMap[pos.x][pos.y]:
                score[i] = -1
            elif state['board'].cells[pos].ship != None and state['board'].cells[pos].ship.player_id != state['me']:
                if state['board'].cells[pos].ship.halite <= s.halite:
                    score[i] = 100000
                else:
                    score[i] += state['board'].cells[pos].ship.halite 
            else:
                score[i] = 5000

            score[i] += state['controlMap'][pos.x][pos.y]

        i, maximum = 0,0 
        for j, thing in enumerate(score):
            if thing > maximum:
                i = j
                maximum = thing
        if maximum < 10:
            return None
        else:
            return direction_to(sPos,get_adjacent(sPos)[i])
    else:
        print("by ally")
        return None






# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell

def get_reward(ship,cell):

    # Don't be stupid
    if state[ship]['blocked'][cell.position.x][cell.position.y] and cell.shipyard == None:
        return 0
    # Mining reward
    elif (cell.ship is None or cell.ship.player_id == state['me']) and cell.shipyard is None:
        return mine_reward(ship,cell)
    elif cell.ship is not None and cell.ship.player_id != state['me']:
        return attack_reward(ship,cell)
    elif cell.shipyard is not None and cell.shipyard.player_id == state['me']:
        return return_reward(ship,cell)
    return 0

def mine_reward(ship,cell):

    mineWeights = weights[1]
    sPos = ship.position
    cPos = cell.position

    # Halite per turn
    halitePerTurn = 0
    if state['currentHalite'] > 1000: # Do we need some funds to do stuff?
        # No
        halitePerTurn = halite_per_turn(cell.halite,dist(sPos,cPos),0) 
    else:
        # Yes
        halitePerTurn = halite_per_turn(cell.halite,dist(sPos,cPos),dist(cPos,state['closestShipyard'][cPos.x][cPos.y]))
    # Surrounding halite
    spreadGain = state['haliteSpread'][cPos.x][cPos.y] * mineWeights[0]
    res = halitePerTurn + spreadGain

    # Penalty 
    if cell.ship != None and not cell.ship is ship:
        res = res / 2

    return res

def attack_reward(ship,cell):

    attackWeights = weights[2]
    cPos = cell.position 
    sPos = ship.position
    d = dist(ship.position,cell.position)
    if cell.ship.halite > ship.halite:
        return (cell.halite + ship.halite) / d**2
    if len(state['myShips']) > 10:
        return state['controlMap'][cPos.x][cPos.y] * 100 / d**2
    return 0

def return_reward(ship,cell):

    returnWeights = weights[3]
    sPos = ship.position
    cPos = cell.position

    if sPos == cPos:
        return 0

    if state['currentHalite'] > 1000:
        return ship.halite / (dist(sPos,cPos)) * 0.1
    else:
        return ship.halite / (dist(sPos,cPos))

# Returns the reward of converting a shipyard in the area.
def shipyard_reward_map():

    N = state['configuration'].size

    closestShipyard = closestShipyard = np.zeros((N,N))
    if len(state['myShipyards']) != 0:
        closestShipyardPosition = state['closestShipyard']
        for x in range(N):
            for y in range(N):
                closestShipyard[x][y] = dist(Point(x,y),closestShipyardPosition[x][y])

    # As we are trying to find the "best" relative, normalizing each with respect 
    # To the maximum element should suffice. 

    closestShipyard = normalize(closestShipyard.flatten())
    haliteSpread = normalize(state['haliteSpread'].flatten())
    halite = normalize(state['haliteMap'].flatten())
    control = normalize(state['controlMap'].flatten())
    controlAlly = control.copy()
    controlAlly[controlAlly<0] = 0
    controlOpponent = control.copy()
    controlOpponent[controlOpponent>0] = 0

    tensorIn = np.array([closestShipyard,haliteSpread,halite,controlOpponent,controlAlly]).T

    # Linear calculation
    # TODO: Improve by converting to a deep NN
    tensorOut = tensorIn @ np.concatenate((np.array([1]),weights[0]))
    res = np.reshape(tensorOut,(N,N))

    return res

def ship_value():
    res = state['haliteMean'] * 0.25 * (state['configuration']['episodeSteps']- 10 - state['board'].step) * weights[4][0]
    res += len(state['ships']) ** 1.5 * weights[4][1]
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

    print(state['shipValue'])

    # Spawn
    spawn_tasks()