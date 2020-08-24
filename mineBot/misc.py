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

def miner_num():
    if state['board'].step < 280:
        if len(state['myShips']) > 25:
            return min(len(state['myShips']),int(state['haliteMean'] / 4 + len(state['myShipyards'])))
        else:
            return min(len(state['myShips']),int(state['haliteMean'] / 2 + len(state['myShipyards'])))
    elif state['board'].step > 370:
        return len(state['myShips'])
    else:
        return len(state['myShips']) * 0.8

