# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell
# This should take the form of a neural network
def getReward(ship,cell):
    global state
    return 1

def naiveReward(ship,cell):
    # For testing purposes
    if cell.ship != None and cell.ship.player_id != state['me'] and cell.ship.halite < ship.halite:
        return -100
    else:
        res = cell.halite - dist(ship.position,cell.position) * 10
        if cell.ship == ship:
            res += 50
        return res