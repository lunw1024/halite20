from encode import *
from agent import *
from dependency import *


# Dense Lattice 1
'''
SCHEMA = [
    [0,0,3,3,0,0,0],
    [0,2,1,1,2,2,0],
    [0,2,1,1,1,1,3],
    [3,1,1,1,1,1,3],
    [3,1,1,1,1,2,0],
    [0,2,2,1,1,2,0],
    [0,0,0,3,3,0,0]
]
SCHEMA_SIZE = [7,7]
'''

# Sparse Lattice 1

SCHEMA = [
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,2,2,0,0,0,0],
    [2,1,1,1,1,1,1,2,0],
    [2,1,1,1,1,1,1,2,0],
    [0,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,0,0],
    [2,1,1,1,1,1,1,2,0],
    [2,1,1,1,1,1,1,2,0],
    [0,0,0,2,2,0,0,0,0]
]
SCHEMA_SIZE = [9,9]

def wall_schema(): 
    # design wall structure according to shipyard pos

    if len(state['myShipyards']) == 0:
        return None

    # find best shipyard
    best = None
    for shipyard in state['myShipyards']:  
        if best == None or state['farmMap'][shipyard.position.x][shipyard.position.y] > state['farmMap'][best.position.x][best.position.y]:
            best = shipyard

    # submit schema into farmMap
    farmMap = np.zeros((state['N'],state['N']))
    sPos = best.position
    for x in range(SCHEMA_SIZE[0]):
        for y in range(SCHEMA_SIZE[1]):
            xx = ((sPos.x - SCHEMA_SIZE[0]//2) + x + 21) % 21
            yy = ((sPos.y - SCHEMA_SIZE[0]//2) + y + 21) % 21
            farmMap[xx][yy] = SCHEMA[x][y]
    return farmMap

def farm(ships):
    global action
    cfg = state['configuration']
    board = state['board']
    me = board.current_player

    # Run when under attack
    shipsToAssign = []
    for ship in ships:
        if ship in action:
            continue 
        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (INF*2+state[ship]['danger'][ship.position.x][ship.position.y], ship, state['closestShipyard'][ship.position.x][ship.position.y])
        if ship in action:
            continue # continue its current action
        shipsToAssign.append(ship)
    

    # Get Targets
    targets = [] # (cell, type)
    farmMap = state['farmSchemaMap']
    if farmMap is None:
        return
    for i in board.cells.values():  # Filter targets
        pos = i.position
        # Wall
        if farmMap[pos.x][pos.y] >= 2:
            targets.append((i,'wall'))
        elif i.shipyard != None and i.shipyard.player_id == state['me']:
            targets.append((i,'guard'))
            for j in range(min(6,len(state['myShips']))):
                targets.append((i,'cell'))
        elif i.halite > 0 and (i.ship is None or i.ship.player_id == state['me']):
            targets.append((i,'cell'))
        elif i.ship != None and i.ship.player_id != state['me'] and farmMap[pos.x][pos.y] == 1:
            targets.append((i,'cell'))
        elif i.shipyard != None and i.shipyard.player_id != state['me'] and farmMap[pos.x][pos.y] == 1:
            targets.append((i,'cell'))

    # Calculate rewards
    rewards = np.zeros((len(shipsToAssign), len(targets)))
    for i, ship in enumerate(shipsToAssign):
        for j, target in enumerate(targets):
            rewards[i, j] = get_farm_reward(ship, target)          

    # Assign rewards
    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True)  # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        task = targets[c]
        if task[1] == 'cell':
            cell = cell = targets[c][0]
            if cell.halite == 0 and cell.shipyard == None and (cell.ship == None or cell.ship.player_id == state['me']):
                action[shipsToAssign[r]] = (0, shipsToAssign[r], targets[c][0].position)
            else:
                action[shipsToAssign[r]] = (rewards[r][c], shipsToAssign[r], targets[c][0].position)
        elif task[1] == 'wall':
            action[shipsToAssign[r]] = (1, shipsToAssign[r], targets[c][0].position)
        elif task[1] == 'guard':
            action[shipsToAssign[r]] = (0, shipsToAssign[r], targets[c][0].position)

def get_farm_reward(ship,target):
    
    cell = target[0]
    res = 0
    # Don't be stupid
    if state[ship]['blocked'][cell.position.x][cell.position.y] and cell.shipyard == None:
        res = 0
    if target[1] == 'wall':
        return wall_reward(ship,cell)
    elif target[1] == 'guard':
        return guard_reward(ship,cell)
    elif target[1] == 'cell':
        if (cell.ship is None or cell.ship.player_id == state['me']) and cell.halite > 0:
            if state['farmSchemaMap'][cell.position.x][cell.position.y] == 1:
                res = farm_reward(ship,cell)
            else:
                res = mine_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id == state['me']:
            res = farm_return_reward(ship,cell)
        elif cell.ship is not None and cell.ship.player_id != state['me']:
            res = farm_attack_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id != state['me']:
            res = farm_attack_reward(ship,cell)
        else:
            print("Shouldn't happen")
    
    return res

def wall_reward(ship,cell):
    sPos = ship.position
    cPos = cell.position

    farmMap = state['farmSchemaMap']

    res = 1000 - dist(sPos,cPos)

    if ship.halite > 0:
        return 0

    # Encourage swap
    if sPos == cPos:
        res = res / 2
        if cell.halite > 0:
            res = 0
    
    swapTarget = None
    for pos in get_adjacent(cPos):
        if farmMap[pos.x][pos.y] == farmMap[cPos.x][cPos.y]:
            swapTarget = pos
    
    if swapTarget == sPos:
        res *= 100
    
    swapTargetCell = state['board'].cells[swapTarget]
    # Check if in sync with 城管 already at position
    if sPos != swapTarget and sPos != cPos:
        # Two allies in place
        if (state['ally'][cPos.x][cPos.y] and cell.ship.halite == 0) and (state['ally'][swapTarget.x][swapTarget.y] and swapTargetCell.ship.halite==0):
            return 0
        # One ally in place
        if (state['ally'][cPos.x][cPos.y] and cell.ship.halite == 0) or (state['ally'][swapTarget.x][swapTarget.y] and swapTargetCell.ship.halite==0):
            ally = None
            if state['ally'][cPos.x][cPos.y] and cell.ship.halite == 0:
                ally = cell.ship
            elif state['ally'][swapTarget.x][swapTarget.y] and swapTargetCell.ship.halite==0:
                ally = swapTargetCell.ship
            if not(dist(ally.position,cPos) % 2 != dist(sPos,cPos) % 2):
                return 0
        # No ships. Do nothing

    return res

def farm_reward(ship,cell):
    # and state['ally'][cell.position.x][cell.position.y]
    if cell.halite < min(500,state['board'].step*2):
        return 0
    return mine_reward(ship,cell)

def farm_attack_reward(ship,cell):
    sPos = ship.position
    cPos = cell.position
    res = 0
    if cell.ship != None:
        if cell.ship.halite > ship.halite:
            res = 200 - dist(sPos,cPos)
        elif cell.ship.halite == ship.halite:
            res = attack_reward(ship,cell)
    elif cell.shipyard != None:
        res = 200 - dist(sPos,cPos)

    if ship.halite > 0:
        res = res / 2
    return res

def farm_return_reward(ship,cell):
    return ship.halite