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
        # Don't mine if it will put ship in danger
        if get_danger(ship.halite+cell.halite*0.25)[cPos.x][cPos.y] > 1:
            return 0
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

    # Danger penalty
    if state[ship]['danger'][cPos.x][cPos.y] > 1:
        res -= mineWeights[3] ** state[ship]['danger'][cPos.x][cPos.y]
        
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


        



        
