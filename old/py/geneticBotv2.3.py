weights='''0.9790460159455175 521.2407434909387 0.7923793908718682 0.7465819541587124 0.15346940177639612 -1.544244770235407
0.02812899270203704 2.446412587257225 0.47149352967640235
0.0934167398237842 229.07285374558384
0.32917669267944993 0.12670831197102922
1.0216033904875512 -3.222779158469551 -2.684281446031774
122.15873770334022'''
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

    # Infinity value thats actually not infinity
INF = 999999999999
    # All game state goes here - everything, even mundane
state = {}

    # Bot training weights
        # 0 - shipyard reward
        # 1 - mine reward
        # 2 - attack weights
        # 3 - return weights
        # 4 - spawn weights
        # 5 - guard weights
    
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
    return closest_thing(t,state['myShips'])

def closest_thing(t,arr):
    res = None
    for thing in arr:
        if res == None:
            res = thing
        elif dist(t,res.position) > dist(t,thing.position):
            res = thing
    return res

def closest_thing_position(t,arr):
    res = None
    for thing in arr:
        if res == None:
            res = thing
        elif dist(t,res) > dist(t,thing):
            res = thing
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
farms = [] # list of cells to farm

def farm_tasks():
    control_farm()
    if len(farms) < 4 * len(state['myShipyards']):
        build_farm()
    # Create patrols

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
                    action[ship] = (INF*2, ship, state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue # continue its current action

        # End-game return
        if board.step > state['configuration']['episodeSteps'] - cfg.size * 2 and ship.halite > 0:
            action[ship] = (ship.halite, ship, state['closestShipyard'][ship.position.x][ship.position.y])
        # End game attack
        if len(state['board'].opponents) > 0 and board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite == 0:
            if len(state['myShipyards']) > 0 and ship == closest_thing(state['myShipyards'][0].position,state['myShips']):
                action[ship] = (0,ship,state['myShipyards'][0].position)
                continue
            killTarget = state['killTarget']
            if len(killTarget.shipyards) > 0:
                target = closest_thing(ship.position,killTarget.shipyards)
                action[ship] = (ship.halite, ship, target.position)
            elif len(killTarget.ships) > 0:
                target = closest_thing(ship.position,killTarget.ships)
                action[ship] = (ship.halite, ship, target.position)

        if ship in action:
            continue

        shipsToAssign.append(ship)

    # Reward based: Attack, Mine
    targets = [] # (cell, type)
    for i in board.cells.values():  # Filter targets
        if i.shipyard != None and i.shipyard.player_id == state['me']:
            targets.append((i,'guard'))
            for j in range(min(6,len(state['myShips']))):
                targets.append((i,'cell'))
            continue
        if i.halite == 0  and i.ship == None and i.shipyard == None:
            # Spots not very interesting
            continue
        targets.append((i,'cell'))

    rewards = np.zeros((len(shipsToAssign), len(targets)))

    for i, ship in enumerate(shipsToAssign):
        for j, target in enumerate(targets):
            rewards[i, j] = get_reward(ship, target)
            
    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True)  # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        task = targets[c]
        if task[1] == 'cell':
            action[shipsToAssign[r]] = (rewards[r][c], shipsToAssign[r], targets[c][0].position)
        elif task[1] == 'guard':
            action[shipsToAssign[r]] = (0, shipsToAssign[r], targets[c][0].position)

    # Process actions
    actions = list(action.values())
    actions.sort(reverse=True, key=lambda x: x[0])
    for act in actions:
        process_action(act)

def process_action(act):
    global action
    if action[act[1]] == True:
        return act[1].next_action
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
        if state['currentHalite'] >= 500 and not state['next'][shipyard.cell.position.x][shipyard.cell.position.y]:
            if state['shipValue'] >= 500:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShips']) < 1 and shipyard == shipyards[0]:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShipyards']) == 1:
                for pos in get_adjacent(shipyard.position):
                    cell = state['board'].cells[pos]
                    if cell.ship != None and cell.ship.player_id != state['me']:
                        shipyard.next_action = ShipyardAction.SPAWN
                        state['currentHalite'] -= 500
                        return

def convert_tasks():
    global action

    # Add convertion tasks

    currentShipyards = state['myShipyards']  # Shipyards "existing"
    targetShipyards = currentShipyards[:]

    # Maximum cell
    v = shipyard_value(state['board'].cells[Point(0,0)])
    t = state['board'].cells[Point(0,0)]
    for cell in state['board'].cells.values():
        a = shipyard_value(cell)
        if v < a:
            v = a
            t = cell
    #print("Shipyard Value",v)
    #print("Ship value", state['shipValue'])
    tx, ty = t.position.x,t.position.y
    # Calculate the reward for each cell
    if state['board'].step == 0:
        # Build immediately
        targetShipyards.append(state['board'].cells[state['myShips'][0].position])
        action[state['myShips'][0]] = (math.inf, state['myShips'][0], state['myShips'][0].position)
        state['currentHalite'] -= 500
    elif len(currentShipyards) == 0:
        # Grab the closest ship to the target and build.
        closest = closest_ship(Point(tx, ty))
        action[closest] = (math.inf, closest, Point(tx, ty))
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500
    elif v > 500 and v > state['shipValue']:
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500

    state['closestShipyard'] = closest_shipyard(targetShipyards)

def build_farm():
    global farms
    maxCell = None
    v = 0

    for cell in state['board'].cells.values():
        if cell.position in farms:
            continue
        a = farm_value(cell)
        if a > v:
            maxCell = cell
            v = a

    if maxCell != None:
        farms.append(maxCell.position)

def control_farm():
    global farms
    for i,farm in enumerate(farms[:]):
        if state['board'].cells[farm].halite < state['haliteMean'] / 1.5 or dist(farm,state['closestShipyard'][farm.x][farm.y]) > 8:
            # Not worth it
            farms.remove(farm)

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
    state['enemyShips'] = []
    for ship in state['ships']:
        state['shipMap'][ship.player_id][ship.position.x][ship.position.y] = 1
        if ship.player_id != state['me']:
            state['enemyShips'].append(ship)
    # Shipyards
    state['shipyardMap'] = np.zeros((state['playerNum'], N, N))
    state['enemyShipyards'] = []
    for shipyard in state['shipyards']:
        state['shipyardMap'][shipyard.player_id][shipyard.position.x][shipyard.position.y] = 1
        if shipyard.player_id != state['me']:
            state['enemyShipyards'].append(shipyard)
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
        state[ship]['danger'] = get_danger(ship)
    # Who we should attack
    if len(state['board'].opponents) > 0:
        state['killTarget'] = get_target()
    
def get_avoidance(s):
    threshold = s.halite
    #Enemy units
    temp = np.where(state['enemyShipHalite'] < threshold, 1, 0)
    enemyBlock = np.copy(temp)
    enemyBlock = enemyBlock + np.roll(temp,1,axis=0)
    enemyBlock = enemyBlock + np.roll(temp,-1,axis=0)
    enemyBlock = enemyBlock + np.roll(temp,1,axis=1)
    enemyBlock = enemyBlock + np.roll(temp,-1,axis=1)

    enemyBlock = enemyBlock + state['enemyShipyard']

    blocked = enemyBlock
    blocked = np.where(blocked>0,1,0)
    return blocked

def get_danger(s):
    threshold = s.halite
    temp = np.where(state['enemyShipHalite'] < threshold, 1, 0)
    dangerMap = np.zeros((temp.shape))
    for i in range(1,4):
        dangerMap += np.roll(temp,i,axis=0) * 0.7**i
        dangerMap += np.roll(temp,-i,axis=0) * 0.7**i
    temp = dangerMap.copy()
    for i in range(1,4):
        dangerMap += np.roll(temp,i,axis=1) * 0.7**i
        dangerMap += np.roll(temp,-i,axis=1) * 0.7**i
    return dangerMap
    

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

def get_target():
    board = state['board']
    me = board.current_player
    idx,v = 0, -math.inf
    for i,opponent in enumerate(board.opponents):
        value = 0
        if opponent.halite-me.halite > 0:
            value = -(opponent.halite-me.halite)
        else:
            value = (opponent.halite-me.halite) * 5
        if value > v:
            v = value
            idx = i
    return board.opponents[idx]

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
        return s.translate(Point(0, 1),N)
    elif d == ShipAction.SOUTH:
        return s.translate(Point(0, -1),N)
    elif d == ShipAction.EAST:
        return s.translate(Point(1, 0),N)
    elif d == ShipAction.WEST:
        return s.translate(Point(-1, 0),N)
    else:
        return s
    
# Returns list of len 4 of adjacent points to a point
def get_adjacent(point):
    N = state['configuration'].size
    res = []
    for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
        res.append(point.translate(Point(offX,offY),N))
    return res
    
def safe_naive(s,t,blocked):
    for direction in directions_to(s.position,t):
        target = dry_move(s.position,direction)
        if not blocked[target.x][target.y]:
            return direction
    return None

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
    # Don't ram stuff thats not the target. Unless we have an excess of ships. Or we are trying to murder a team.
    if len(state['myShips']) < 30 and state['board'].step < state['configuration']['episodeSteps'] - state['configuration'].size * 1.5:
        blocked += np.where(state['enemyShipHalite'] == s.halite, 1, 0)

    blocked = np.where(blocked>0,1,0)

    desired = None

    #Stay still
    if sPos == t or nextMap[t.x][t.y]:

        #Someone with higher priority needs position, must move. Or being attacked.
        if blocked[t.x][t.y]:
            for processPoint in get_adjacent(sPos):
                if not blocked[processPoint.x][processPoint.y]:
                    nextMap[processPoint.x][processPoint.y] = 1
                    desired = direction_to(sPos,processPoint)
                    t = processPoint
            if desired == None:
                target = micro_run(s)
                t = dry_move(sPos,target)
                desired = target
        else:
            t = sPos
            desired = None
    else:
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

            # Can go in general direction
            res = safe_naive(s,t,blocked)
            if res != None:
                t = dry_move(s.position,res)
                desired = res
            else:
                #Random move
                for processPoint in get_adjacent(sPos):
                    if not blocked[processPoint.x][processPoint.y]:
                        nextMap[processPoint.x][processPoint.y] = 1
                        t = processPoint
                        desired = direction_to(sPos,processPoint)
                
                # Run
                if desired == None and blocked[sPos.x][sPos.y]:
                    target = micro_run(s)
                    t = dry_move(sPos,target)
                    desired = target
                elif not blocked[sPos.x][sPos.y]:
                    t = sPos
                    desired = None        
        else:
            # Path reconstruction
            while pred[t] != sPos:
                t = pred[t]

            desired = direction_to(sPos,t)

    # Reduce collisions
    if desired != None and state['board'].cells[t].ship != None and state['board'].cells[t].ship.player_id == state['me']:
        target = state['board'].cells[t].ship
        s.next_action = desired
        if action[target] != True:
            nextMap[t.x][t.y] = 1
            result = process_action(action[target])
            # Going there will kill it
            if result == None:
                desired = a_move(s,t,inBlocked)
                nextMap[t.x][t.y] = 0
                t = dry_move(sPos,desired)
    nextMap[t.x][t.y] = 1
    return desired

# Ship might die, RUN!
def micro_run(s):
    sPos = s.position
    nextMap = state['next']

    if state[s]['blocked'][sPos.x][sPos.y]:
        if s.halite > 500:
            return ShipAction.CONVERT
        score = [0,0,0,0]
        for i,pos in enumerate(get_adjacent(sPos)):
            if nextMap[pos.x][pos.y]:
                score[i] = -1
            elif state['board'].cells[pos].ship != None and state['board'].cells[pos].ship.player_id != state['me']:
                if state['board'].cells[pos].ship.halite >= s.halite:
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
        return None






# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell

def get_reward(ship,target):
    
    cell = target[0]
    res = 0
    # Don't be stupid
    if state[ship]['blocked'][cell.position.x][cell.position.y] and cell.shipyard == None:
        res = 0
    elif target[1] == 'cell':
        # Mining reward
        if (cell.ship is None or cell.ship.player_id == state['me']) and cell.halite > 0:
            res = mine_reward(ship,cell)
        elif cell.ship is not None and cell.ship.player_id != state['me']:
            res = attack_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id == state['me']:
            res = return_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id != state['me']:
            res = attack_reward(ship,cell)
    elif target[1] == 'guard':
        res = guard_reward(ship,cell)
    return res

def guard_reward(ship,cell):
    cPos = cell.position
    sPos = ship.position
    guardWeights = weights[5]
    if len(state['enemyShips']) == 0:
        return 0
    closestEnemy = closest_thing(ship.position,state['enemyShips'])
    if dist(sPos,cPos) > dist(closestEnemy.position,cPos):
        return 0
    elif ship.halite != 0 and dist(sPos,cPos) >= dist(closestEnemy.position,cPos):
        return 0

    # Check if we want to build
    if cell.shipyard == max(state['myShipyards'],key=lambda shipyard: state['haliteSpread'][shipyard.position.x][shipyard.position.y]):
        if state['currentHalite'] >= 500 and state['shipValue'] > 500:
            return 0
    
    return guardWeights[0] / (dist(closestEnemy.position,cPos) * (dist(sPos,cPos)+1))
 
def mine_reward(ship,cell):

    mineWeights = weights[1]
    sPos = ship.position
    cPos = cell.position
    cHalite = cell.halite

    # Halite per turn
    halitePerTurn = 0
    
    # Current cell
    if sPos == cPos:
        # Current cell multiplier
        if cHalite > state['haliteMean'] * mineWeights[2] and ship.halite > 0:
            cHalite = cHalite * mineWeights[1]
        # Farming!
        if cPos in farms and cell.halite < min(500,(state['board'].step + 10*15)) and state['board'].step < state['configuration']['episodeSteps'] - 50:
            return 0
        # Don't mine if enemy near
        for pos in get_adjacent(sPos):
            if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
                return 0
    
    # Nearby 
    if cPos in get_adjacent(sPos) and state['controlMap'][cPos.x][cPos.y] < 0.5:
        # Try to reduce collision num
        for pos in get_adjacent(cPos):
            if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
                return 0
    '''
    if state['currentHalite'] > 1000: # Do we need some funds to do stuff?
        # No
        halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),0) 
    else:
        # Yes
        halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),dist(cPos,state['closestShipyard'][cPos.x][cPos.y]))
    '''
    halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),dist(cPos,state['closestShipyard'][cPos.x][cPos.y])) 
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
    multiplier = 1
    
    # Don't even bother
    if dist(sPos,cPos) > 4:
        return 0

    # Defend the farm!
    if cPos in farms:
        return cell.halite - d

    res = 0
    # It's a ship!
    if cell.ship != None:

            # Nearby 
        if cPos in get_adjacent(sPos) and state['controlMap'][cPos.x][cPos.y] < 0.5:
            # Try to reduce collision num
            for pos in get_adjacent(cPos):
                if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite and cell.ship.player != state['board'].cells[pos].ship.player:
                    return 0

        if cell.ship.halite > ship.halite:
            res = max(cell.halite / d,state['controlMap'][cPos.x][cPos.y] * 100) / d**2
        elif len(state['myShips']) > 10:
            res = state['controlMap'][cPos.x][cPos.y] * 100 / d**2
        if ship.halite != 0:
            res = res / 3
    
    # It's a shipyard!
    elif len(state['myShips']) > 10:
        if len(state['myShips']) > 15 and cell.shipyard.player == state['killTarget']:
            # Is it viable to attack
            viable = True
            for pos in get_adjacent(cPos):
                target = state['board'].cells[pos].ship
                if target != None and target.player_id != state['me'] and target.halite <= ship.halite:
                    viable = False
                    break
            if viable:
                res = attackWeights[1] / d**2
        
        res = max(res,state['controlMap'][cPos.x][cPos.y] * 100 / d**2)

    return res * attackWeights[0]

def return_reward(ship,cell):

    returnWeights = weights[3]
    sPos = ship.position
    cPos = cell.position

    if sPos == cPos :
        return 0
    res = 0
    if state['currentHalite'] > 1000:
        res = ship.halite / (dist(sPos,cPos)) * returnWeights[0]
    else:
        res = ship.halite / (dist(sPos,cPos))
     
    res = res * returnWeights[1]
    return res 

def shipyard_value(cell):
    # Features
    shipyardWeights = weights[0]
    cPos = cell.position

    nearestShipyard = closest_thing(cPos,state['shipyards'])
    nearestShipyardDistance = 1
    if nearestShipyard != None:
        nearestShipyardDistance = dist(nearestShipyard.position,cPos)
    negativeControl = min(0,state['controlMap'][cPos.x][cPos.y])
    if len(state['myShips']) > 0:
        negativeControl -= 0.5 ** dist(closest_thing(cPos,state['myShips']).position,cPos)
    haliteSpread = state['haliteSpread'][cPos.x][cPos.y] - state['haliteMap'][cPos.x][cPos.y]
    shipShipyardRatio = len(state['myShips']) / max(1,len(state['myShipyards']))

    # Hard limit on range and halite spread
    if nearestShipyardDistance <= 5 or haliteSpread <= 200:
        return 0

    # Base halite multiplier
    res = haliteSpread * shipyardWeights[0]

    # Negative control
    res += negativeControl * shipyardWeights[1]

    # Nearest shipyard
    res = res * nearestShipyardDistance ** shipyardWeights[2]

    # Ship shipyard ratio multiplier
    res = res * shipShipyardRatio ** shipyardWeights[3]

    # Final multiplier and bias
    res = res * shipyardWeights[4] + shipyardWeights[5]

    return res

def ship_value():
    res = state['haliteMean'] * 0.25 * (state['configuration']['episodeSteps']- 30 - state['board'].step) * weights[4][0]
    res += (len(state['ships']) - len(state['myShips'])) ** 1.5 * weights[4][1]
    res += len(state['myShips'])  ** 1.5 * weights[4][2]
    return res

def farm_value(cell):
    cPos = cell.position
    if len(state['myShipyards']) == 0 or cell.halite == 0:
        return 0

    closest = state['closestShipyard'][cPos.x][cPos.y]
    if dist(closest,cPos) <= 1 or dist(closest,cPos) > 4:
        return 0

    return (cell.halite**0.5) / dist(closest,cPos) ** 2


        



        

# The final function


@board_agent
def agent(board):

    # Init
    if board.step == 0:
        init(board)

    # Update
    update(board)

    # Convert
    convert_tasks()

    # Farm
    #farm_tasks()

    # Ship
    ship_tasks()

    # Spawn
    spawn_tasks()