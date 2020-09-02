from dependency import *
from navigation import *

# Core strategy

action = {}  # ship -> (value,ship,target)

def ship_tasks():  # update action
    global action
    cfg = state["configuration"]
    board = state["board"]
    me = board.current_player

    # Assign roles
    state['farmers'] = [] # who fight and consist the wall
    state['miners'] = []
    state['endgame'] = []

    # Select strategy

    if board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5:
        end_game()
    #TODO Add farming decider:
    else:
        early_game()

    # Calculate actions
    mine(state['miners'])
    farm(state['farmers'])
    endgame(state['endgame'])


    # Process actions
    actions = list(action.values())
    actions.sort(reverse=True, key=lambda x: x[0])
    for act in actions:
        process_action(act)

def process_action(act):
    global action
    if action[act[1]] == True:
        return act[1].next_action
    action[act[1]] = True
    # Processing
    act[1].next_action = d_move(act[1], act[2], state[act[1]]["blocked"])
    # Ship convertion
    sPos = act[1].position
    if state["closestShipyard"][sPos.x][sPos.y] == sPos and state["board"].cells[sPos].shipyard == None:
        act[1].next_action = ShipAction.CONVERT
        state["next"][sPos.x][sPos.y] = 1
    return act[1].next_action

def early_game():
    global action
    me = state['board'].current_player
    miners = state['miners']
    for ship in me.ships:
        if ship in action:
            continue
        miners.append(ship)

def mid_game():
    me = state['board'].current_player
    schema = wall_schema()
    vacancy = np.sum(schema > 0)
    farmers = state['farmers']
    miners = state['miners']
    for ship in me.ships: # hard rules
        x, y = ship.position
        if ship.halite > 0:
            miners.append(ship)
        elif schema[x, y] > 0: # stay
            farmers.append(ship)
            vacancy -= 1
    for ship in me.ships: # the rest
        if ship in miners or ship in farmers:
            continue
        if vacancy > 0 and ship.halite == 0: # fill wall vacancy
            farmers.append(ship)
            vacancy -= 1
        else:
            miners.append(ship) # TODO: currently excessive ship are all miners, consider adding "colonizer"

def end_game():
    global action
    me = state['board'].current_player
    endgame = state['endgame']
    for ship in me.ships:
        if ship in action:
            continue
        endgame.append(ship)
