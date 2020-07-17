# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell
# This should take the form of a neural network
def get_reward(ship,cell):

    #Don't be stupid
    if state[ship]['blocked'][cell.position.x][cell.position.y] and cell.shipyard != None:
        return 0
    #Mining reward
    elif (cell.ship is None or cell.ship.player_id==state['me']) and cell.shipyard is None:
        return mine_reward(ship,cell)
    elif cell.ship is not None and not cell.ship.player_id==state['me']:
        return attack_reward(ship,cell)
    elif cell.shipyard is not None and cell.shipyard.player_id==state['me']:
        return return_reward(ship,cell)
    else: 
        pass
    return 0

def mine_reward(ship,cell):

    sPos = ship.position
    cPos = cell.position

    # Features
        # Halite per turn - weight constant
        # Control map 
            # Positive
            # Negative
        # Halite spread

        # As majority of features are shared, see process.py mine_value_map()

    halitePerTurn = halite_per_turn(ship.halite,cell.halite,dist(sPos,cPos),dist(cPos,state['closestShipyard'][cPos.x][cPos.y]))

    return halitePerTurn + state['mineValueMap'][cPos.x][cPos.y]

def attack_reward(ship,cell):

    cPos = cell.position 
    sPos = ship.position

    # Features
        # One hot - is ship empty
        # Distance from current ship
        # Control map
        # Halite spread on target ship

    # Currently just a placeholder
    
    return max(cell.halite,state['controlMap'][cPos.x][cPos.y]*100) / dist(ship.position,cell.position)**2 * weights[2][0]

def return_reward(ship,cell):

    cPos = cell.position

    # Features
        # Halite on ship
        # Halite spread
        # Halite
        # Distance
        # Ship creation value
    # Currently just a placeholder
    return (ship.halite - cell.halite) - weights[3][0] * state['haliteMean']

# As this is a linear function, and most features are ship independant
# This should speed things up
def mine_value_map():
    # The divisions are to lower the values to around 1. This is so we can utilize the 
    # same step change while training
    N = state['configuration'].size
    halite = state['haliteMap'].flatten() / 500 
    spread = state['haliteSpread'].flatten() / 2000
    control = state['controlMap'].flatten() / 4
    controlAlly = control.copy()
    controlAlly[controlAlly<0] = 0
    controlOpponent = control.copy()
    controlOpponent[controlOpponent>0] = 0
    tensorIn = np.array([halite,controlAlly,controlOpponent,spread]).T
    tensorOut = tensorIn @ weights[1]
    res = np.reshape(tensorOut,(N,N))
    return res

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
    tensorOut = tensorIn @ weights[0]
    res = np.reshape(tensorOut,(N,N))

    return res



        
