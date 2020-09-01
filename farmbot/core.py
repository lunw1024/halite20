from dependency import *
from navigation import *

# Core strategy

action = {}  # ship -> (value,ship,target)

# XXX
SCHEMA = [((-1, 3), (0, 3)), ((1, 2), (2, 2)),
          ((-1, -3), (0, -3)), ((1, -2), (2, -2)),
          ((1, 3), (0, 3)), ((-1, 2), (-2, 2)),
          ((1, -3), (0, -3)), ((-1, -2), (-2, -2))]


def wall_schema(): 
    # design wall structure according to shipyard pos
    # out[i, j] > 0 means (i, j) should be wall, the value indicates the pair (to swap non-stop)
    # TODO: remove connected walls
    cfg = state["configuration"]
    board = state["board"]
    me = board.current_player
    N = cfg.size
    out = np.zeros((N, N), dtype=int)
    for shipyard in me.shipyards:
        x, y = shipyard.position
        for i, ((dx1, dy1), (dx2, dy2)) in enumerate(SCHEMA):
            out[(x + dx1) % N, (y + dy1) % N] = i + 1
            out[(x + dx2) % N, (y + dy2) % N] = i + 1
    return out


def ship_tasks():  # update action
    global action
    cfg = state["configuration"]
    board = state["board"]
    me = board.current_player
    schema = wall_schema()

    # assign roles
    attackers = [] # who fight and consist the wall
    miners = []
    vacancy = np.sum(schema > 0)
    for ship in me.ships: # hard rules
        x, y = ship.position
        if ship.halite > 0:
            miners.append(ship)
        elif schema[x, y] > 0: # stay
            attackers.append(ship)
            vacancy -= 1
    for ship in me.ships: # the rest
        if ship in miners or ship in attackers:
            continue
        if vacancy > 0 and ship.halite == 0: # fill wall vacancy
            attackers.append(ship)
            vacancy -= 1
        else:
            miners.append(ship) # TODO: currently excessive ship are all miners, consider adding "colonizer"


    # core logic
    for ship in me.ships:
        # TODO

        # attackers (wall builder) swap places in pairs

        # miners mine inside the wall

        # TODO: expand by building more shipyards or increase wall size
        pass

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
