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
        elif dist(t,res.point) > dist(t,ship.point):
            res = ship
    return res