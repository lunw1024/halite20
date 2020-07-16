# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell
# This should take the form of a neural network
def get_reward(ship,cell):
    return optimus_reward(ship,cell)

def optimus_reward(ship,cell):
    reward = 0
    INF = int(1e9)
    me = state['board'].current_player
    if (cell.ship is None or cell.ship is ship) and cell.shipyard is None: # mineral
        d1 = dist(ship.position, cell.position)
        d2 = dist(cell.position, state['closestShipyard'][cell.position.x][cell.position.y]) # TODO: edge case no shipyards
        reward = halite_per_turn(ship.halite, cell.halite, d1 + d2)
    elif cell.ship is not None and cell.ship.player.is_current_player: # friendly ship
        reward = -INF # avoid clustering
    elif cell.ship is not None and not cell.ship.player.is_current_player: # enemy ship
        dist_ = dist(ship.position, cell.position)
        reward = cell.ship.halite / (dist_ * CHASE_PUNISHMENT) if cell.ship.halite > me.halite and dist_ <= MAX_CHASE_RANGE else 0 # TUNABLE
    elif cell.shipyard is not None and cell.shipyard.player.is_current_player: # friendly shipyard
        reward = ship.halite / max(dist(ship.position, cell.position), 0.1) # TODO: TUNABLE?
    elif cell.shipyard is not None and not cell.shipyard.player.is_current_player: # enemy shipyard
        reward = SHIPYARD_DEMOLISH_REWARD / dist(ship.position, cell.position)
    return reward

def naive_reward(ship,cell):
    # For testing purposes
    if cell.ship != None and cell.ship.player_id != state['me'] and cell.ship.halite < ship.halite:
        return -100
    else:
        res = cell.halite - dist(ship.position,cell.position) * 10
        if cell.ship == ship:
            res += 50
        return res

# Returns the reward of converting a shipyard in the area, relative.
# TODO: Convert to absolute instead of relative

def shipyard_reward_map():

    N = state['configuration'].size

    closestShipyard = closestShipyard = np.zeros((N,N))
    if len(state['myShipyards']) != 0:
        closestShipyardPosition = state['closestShipyard']
        for x in range(N):
            for y in range(N):
                closestShipyard[x][y] = dist(Point(x,y),closestShipyardPosition[x][y])

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



        
