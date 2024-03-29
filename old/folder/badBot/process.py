# General calculations whose values are expected to be used in multiple instances
# Basically calc in botv1.0. 
# Run in update() - see dependency.py
from dependency import *
from navigation import *

def encode():
    global state
    
    N = state['configuration'].size

    # Halite 
    state['haliteMap'] = np.zeros((N, N))
    for cell in state['cells']:
        state['haliteMap'][cell.position.x][cell.position.y] = cell.halite
    # Halite Spread
    state['haliteSpread'] = np.copy(state['haliteMap'])
    for i in range(3):
        state['haliteSpread'] += np.roll(state['haliteMap'],i,axis=0) / (i+1)
        state['haliteSpread'] += np.roll(state['haliteMap'],-i,axis=0) / (i+1)
    temp = state['haliteSpread'].copy()
    for i in range(3):
        state['haliteSpread'] += np.roll(temp,i,axis=1) / (i+1)
        state['haliteSpread'] += np.roll(temp,-i,axis=1) / (i+1)
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
