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
    elif cell.shipyard is not None and cell.shipyard.player_id != state['me']:
        return attack_reward(ship,cell)
    return 0

def mine_reward(ship,cell):

    mineWeights = weights[1]
    sPos = ship.position
    cPos = cell.position
    cHalite = cell.halite

    # Halite per turn
    halitePerTurn = 0

    # Farming!
    if cPos in farms and cell.halite < min(500,(state['board'].step + 10*15)):
        return -1
 
    # Multiplier to current cell
    if sPos == cPos and cHalite > state['haliteMean'] / 2:
        cHalite = cHalite * mineWeights[1]

    if state['currentHalite'] > 1000: # Do we need some funds to do stuff?
        # No
        halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),0) 
    else:
        # Yes
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
        if cell.ship.halite > ship.halite:
            res = (cell.halite + ship.halite) / d**2
        if len(state['myShips']) > 10:
            res = state['controlMap'][cPos.x][cPos.y] * 100 / d**2
    
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

def farm_value(cell):
    cPos = cell.position
    if len(state['myShipyards']) == 0 or cell.halite == 0:
        return 0

    closest = state['closestShipyard'][cPos.x][cPos.y]
    if dist(closest,cPos) <= 1 or dist(closest,cPos) > 4:
        return 0

    return (cell.halite**0.5) / dist(closest,cPos) ** 2


        



        
