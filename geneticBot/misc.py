# General random helper functions that are not strictly "process" or in "nav"

# Map from 0 to 1
def normalize(v):
    norm = np.linalg.norm(v,np.inf)
    if norm == 0: 
       return v
    return v / norm

def closest_ship(t):
    res = None
    for ship in state['myShips']:
        if res == None:
            res = ship
        elif dist(t,res.position) > dist(t,ship.position):
            res = ship
    return res


# optimus_mine helpers

MAX_CHASE_RANGE = 2
CHASE_PUNISHMENT = 2
SHIPYARD_DEMOLISH_REWARD = 700

OPTIMAL_MINING_TURNS = np.array( # optimal mining turn for [Cargo/Deposit, travelTime]
  [[0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
   [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
   [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
   [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
   [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
   [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
   [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

def num_turns_to_mine(C, D, travelTime, minMineTurns=1): # https://www.kaggle.com/krishnaharish/optimus-mine-agent
    # How many turns should we plan on mining?
    # C = carried halite, D = halite deposit, travelTime = steps to square and back to shipyard
    travelTime = int(np.clip(travelTime, 0, OPTIMAL_MINING_TURNS.shape[1] - 1))
    if C == 0:
        cdRatio = 0
    elif D == 0:
        cdRatio = OPTIMAL_MINING_TURNS.shape[0] - 1
    else:
        cdRatio = np.clip(int(math.log(C/D)*2.5+5.5), 0, OPTIMAL_MINING_TURNS.shape[0] - 1)
    return max(OPTIMAL_MINING_TURNS[cdRatio, travelTime], minMineTurns)

def halite_per_turn(cargo, deposit, shipTime, returnTime, minMineTurns=1):
    travelTime = shipTime + returnTime
    turns = num_turns_to_mine(cargo, deposit, travelTime, minMineTurns)
    actualDeposit = max(500,deposit + 1.02 ** shipTime)
    mined = cargo + (1 - .75**turns) * actualDeposit
    return mined / (travelTime + turns)