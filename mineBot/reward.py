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

    return 0
    
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
    cell
    shipyardDist = dist(cPos,state['closestShipyard'][cPos.x][cPos.y])

    if state['generalDangerMap'][cPos.x][cPos.y] > 1.5 and state['trapped'][state['me']][cPos.x][cPos.y]:
        return 0

    # Halite per turn
    halitePerTurn = 0

    # Occupied cell
    if cell.ship != None and cell.ship.player_id == state['me'] and cell.ship.halite <= ship.halite:
        # Current cell multiplier
        if sPos == cPos:
            if cHalite > state['haliteMean'] * mineWeights[2] and cHalite > 10 and ship.halite > 0:
                cHalite = cHalite * mineWeights[1]

        # Farming!
        if cPos in farms and cell.halite < min(500,(state['board'].step + 10*15)) and state['board'].step < state['configuration']['episodeSteps'] - 50:
            return 0
        
        if shipyardDist >= 3:
            # Don't mine if enemy near
            for pos in get_adjacent(cPos):
                if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
                    return 0
            
            if state['trapped'][state['me']][cPos.x][cPos.y]:
                return 0
    
    # Dangerous area
    cHalite += state['negativeControlMap'][cPos.x][cPos.y] * mineWeights[4]
    
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
            # Defend the farm!
            if cPos in farms:
                return cell.halite - d
            res = max([cell.halite**(attackWeights[4]),state['controlMap'][cPos.x][cPos.y]*attackWeights[2]]) - d*attackWeights[3]
        elif len(state['myShips']) > 15:
            res = state['controlMap'][cPos.x][cPos.y] * 100 / d**2
        if ship.halite != 0:
            res = res / 3
    
    # It's a shipyard!
    elif len(state['myShips']) > 10 and ship.halite == 0:
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

    if state['board'].step > 310:
        return 0

    nearestShipyard = closest_thing(cPos,state['shipyards'])
    nearestShipyardDistance = 1
    if nearestShipyard != None:
        nearestShipyardDistance = dist(nearestShipyard.position,cPos)
    negativeControl = min(0,state['controlMap'][cPos.x][cPos.y])
    if len(state['myShips']) > 0:
        negativeControl = max(negativeControl-0.5 ** dist(closest_thing(cPos,state['myShips']).position,cPos),state['negativeControlMap'][cPos.x][cPos.y])
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
    if len(state['myShips']) >= 60:
        return 0
    res = state['haliteMean'] * 0.25 * (state['configuration']['episodeSteps']- 30 - state['board'].step) * weights[4][0]
    res += (len(state['ships']) - len(state['myShips'])) ** 1.5 * weights[4][1]
    res += len(state['myShips'])  ** 1.5 * weights[4][2]
    return res 
        



        
