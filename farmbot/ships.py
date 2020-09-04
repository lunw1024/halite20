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

    if board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and len(state["board"].opponents) > 0:
        end_game()
    elif len(me.ships) > 18 and board.step < 300:
        #TODO: Improve decider
        mid_game()
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
    farmers = state['farmers']
    if len(me.ships) > 18:
        for ship in me.ships:
            if ship in action:
                continue
            farmers.append(ship)

    else:
        early_game()

def end_game():
    global action
    me = state['board'].current_player
    endgame = state['endgame']
    for ship in me.ships:
        if ship in action:
            continue
        endgame.append(ship)
