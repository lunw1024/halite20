from dependency import *
from navigation import *

# Core strategy

action = {}  # ship -> (value,ship,target)
farms = [] # list of cells to farm

def ship_tasks():  # update action
    global action
    cfg = state['configuration']
    board = state['board']
    me = board.current_player
    tasks = {}
    shipsToAssign = []

    # Rule based: Flee, Return
    for ship in me.ships:
        # Flee
        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (math.inf, ship, state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue # continue its current action

        # End-game return
        if board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite > 0:
            action[ship] = (ship.halite, ship, state['closestShipyard'][ship.position.x][ship.position.y])
            
        if ship in action:
            continue

        # TODO: Force attack

        shipsToAssign.append(ship)

    # Reward based: Attack, Mine
    targets = []
    for i in board.cells.values():  # Filter targets
        if i.shipyard != None and i.shipyard.player_id == state['me']:
            for j in range(min(8,len(state['myShips']))):
                targets.append(i)
            continue
        if i.halite == 0  and i.ship == None:
            continue
        targets.append(i)
    rewards = np.zeros((len(shipsToAssign), len(targets)))

    for i, ship in enumerate(shipsToAssign):
        for j, cell in enumerate(targets):
            rewards[i, j] = get_reward(ship, cell)

    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True)  # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        action[shipsToAssign[r]] = (rewards[r][c], shipsToAssign[r], targets[c].position)

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
    act[1].next_action = a_move(act[1], act[2], state[act[1]]['blocked'])
    # Ship convertion
    sPos = act[1].position
    if state['closestShipyard'][sPos.x][sPos.y] == sPos and state['board'].cells[sPos].shipyard == None:
        act[1].next_action = ShipAction.CONVERT
        
    return act[1].next_action

def spawn_tasks():
    shipyards = state['board'].current_player.shipyards
    shipyards.sort(reverse=True, key=lambda shipyard: state['haliteSpread'][shipyard.position.x][shipyard.position.y])
    for shipyard in shipyards:
        if state['currentHalite'] > 500 and not state['next'][shipyard.cell.position.x][shipyard.cell.position.y]:
            if state['shipValue'] > 500:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShips']) <= 2:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShipyards']) == 1:
                for pos in get_adjacent(shipyard.position):
                    cell = state['board'].cells[pos]
                    if cell.ship != None and cell.ship.player_id != state['me']:
                        shipyard.next_action = ShipyardAction.SPAWN
                        state['currentHalite'] -= 500
                        return


def convert_tasks():
    global action

    # Add convertion tasks

    rewardMap = shipyard_reward_map()  # Best area to build a shipyard
    currentShipyards = state['myShipyards']  # Shipyards "existing"
    targetShipyards = currentShipyards[:]

    t = np.where(rewardMap == np.amax(rewardMap))
    tx, ty = list(zip(t[0], t[1]))[0]

    # Calculate the reward for each cell
    if state['board'].step == 0:
        # Build immediately
        targetShipyards.append(state['board'].cells[state['myShips'][0].position])
        action[state['myShips'][0]] = (math.inf, state['myShips'][0], state['myShips'][0].position)
        state['currentHalite'] -= 500
    elif len(currentShipyards) == 0:
        # Grab the closest ship to the target and build.
        closest = closest_ship(Point(tx, ty))
        action[closest] = (math.inf, closest, Point(tx, ty))
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500
    elif len(state['myShips']) >= len(currentShipyards) * 5 + 6 and len(state['myShipyards']) < 4:
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500

    state['closestShipyard'] = closest_shipyard(targetShipyards)

def build_farm():
    global farms
    maxCell = None
    v = 0

    for cell in state['board'].cells.values():
        if cell.position in farms:
            continue
        a = farm_value(cell)
        if a > v:
            maxCell = cell
            v = a

    if maxCell != None:
        farms.append(maxCell.position)

def control_farm():
    global farms
    for i,farm in enumerate(farms[:]):
        if state['board'].cells[farm].halite < state['haliteMean'] / 1.5:
            # Not worth it
            farms.remove(farm)
