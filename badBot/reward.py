# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell
# This should take the form of a neural network
def getReward(ship,cell):
    global state
    return 1