def mine(ships):
    global action
    cfg = state['configuration']
    board = state['board']
    me = board.current_player

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
    for i in board.cells.values():  # Filter targets
        if i.shipyard != None and i.shipyard.player_id == state['me']:
            targets.append((i,'guard'))
            for j in range(min(6,len(state['myShips']))):
                targets.append((i,'cell'))
            continue
        '''if i.halite < 15 and i.ship == None and i.shipyard == None:
            # Spots not very interesting
            continue'''
        if i.ship != None and i.ship.player_id != state['me']:
            if i.ship.halite == 0 and state['controlMap'][i.position.x][i.position.y] < 0:
                continue
        targets.append((i,'cell'))

    # Calculate rewards
    rewards = np.zeros((len(shipsToAssign), len(targets)))
    for i, ship in enumerate(shipsToAssign):
        for j, target in enumerate(targets):
            rewards[i, j] = get_reward(ship, target)          

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
        elif task[1] == 'guard':
            action[shipsToAssign[r]] = (0, shipsToAssign[r], targets[c][0].position)

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
        elif cell.shipyard is None and cell.halite == 0 and (cell.ship is None or cell.ship.player_id == state['me']):
            res = control_reward(ship,cell)
        elif cell.ship is not None and cell.ship.player_id != state['me']:
            res = attack_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id == state['me']:
            res = return_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id != state['me']:
            res = attack_reward(ship,cell)
    elif target[1] == 'guard':
        res = guard_reward(ship,cell)
    return res

def control_reward(ship,cell):
    
    sPos = ship.position
    cPos = cell.position

    if ship.halite > 0 or dist(cPos,state['closestShipyard'][cPos.x][cPos.y]) <= 2:
        return 0
    res = 0
    for pos in get_adjacent(cPos):
        tCell = state['board'].cells[pos]
        if tCell.halite > 0:
            res += 3.5
    res -= dist(sPos,cPos) + dist(cPos,state['closestShipyard'][cPos.x][cPos.y])
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
        if state['currentHalite'] >= 500 and state['spawn']:
            return 0
    
    return guardWeights[0] / (dist(closestEnemy.position,cPos) * max(dist(sPos,cPos),1))
 
def mine_reward(ship,cell):

    mineWeights = weights[1]
    sPos = ship.position
    cPos = cell.position
    cHalite = cell.halite
    shipyardDist = dist(cPos,state['closestShipyard'][cPos.x][cPos.y])

    if state['generalDangerMap'][cPos.x][cPos.y] > 1.5 and state['trapped'][state['me']][cPos.x][cPos.y]:
        return 0

    # Halite per turn
    halitePerTurn = 0

    '''
    if shipyardDist <= 3:
        if cell.halite < min(100,(state['board'].step + 5*10)) and state['board'].step < state['configuration']['episodeSteps'] - 50:
            return 0
    '''
    
    # Occupied cell
    if cell.ship != None and cell.ship.player_id == state['me'] and cell.ship.halite <= ship.halite:
        # Current cell multiplier
        if sPos == cPos:
            if cHalite > state['haliteMean'] * mineWeights[2] and cHalite > 10 and ship.halite > 0:
                cHalite = cHalite * mineWeights[1]
        
        if shipyardDist >= 3:
            # Don't mine if enemy near
            for pos in get_adjacent(cPos):
                if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
                    return 0
            
            if state['trapped'][state['me']][cPos.x][cPos.y]:
                return 0
    
    # Dangerous area
    #cHalite += state['negativeControlMap'][cPos.x][cPos.y] * mineWeights[4]
    
    if state['enemyShipHalite'][cPos.x][cPos.y] <= ship.halite:
        return 0
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
    
    halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),shipyardDist) 
    # Surrounding halite
    spreadGain = state['haliteSpread'][cPos.x][cPos.y] * mineWeights[0]
    res = halitePerTurn + spreadGain

    if state[ship]['danger'][cPos.x][cPos.y] > 1.3:
        res -= mineWeights[3] ** state[ship]['danger'][cPos.x][cPos.y]
        
    return res

def attack_reward(ship,cell):

    attackWeights = weights[2]
    cPos = cell.position 
    sPos = ship.position
    d = dist(ship.position,cell.position)
    
    # Don't even bother
    if dist(sPos,cPos) > 6:
        return 0

    res = 0
    # It's a ship!
    if cell.ship != None:
            # Nearby 
        if cPos in get_adjacent(sPos) and state['controlMap'][cPos.x][cPos.y] < 0.5:
            # Try to reduce collision num
            for pos in get_adjacent(cPos):
                if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
                    return 0

        if cell.ship.halite > ship.halite:
            res = max([cell.halite**(attackWeights[4]),state['controlMap'][cPos.x][cPos.y]*attackWeights[2]]) - d*attackWeights[3]
        elif len(state['myShips']) > 15:
            res = state['controlMap'][cPos.x][cPos.y] * 100 / d**2
        if ship.halite != 0:
            res = res / 3
    
    # It's a shipyard!
    elif len(state['myShips']) > 10 and ship.halite == 0:
        if len(state['myShips']) > 15 and cell.shipyard.player == state['killTarget'] and dist(sPos,state['closestShipyard'][sPos.x][sPos.y]) <= 2:
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
        