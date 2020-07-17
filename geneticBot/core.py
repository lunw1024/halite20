from dependency import *
from navigation import *

# Core strategy

action = {} #ship -> [value,ship,target]

def ship_tasks(): # return updated tasks
    global action
    cfg = state['configuration']
    board = state['board']
    me = board.current_player
    tasks = {}
    assign = []

    # Rule based
        # Run, return

    for ship in me.ships:
        # Run 
        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (math.inf,ship,state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue

        # Return
        if board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite > 0:
            action[ship] = (ship.halite,ship,state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue

        # TODO: Force attack
            
        assign.append(ship)

    # Reward based
        # Attack, Mine
    targets = []
    for i in board.cells.values(): # Filter targets
        if i.shipyard != None and i.shipyard.player_id == state['me']:
            for j in range(len(state['myShips'])):
                targets.append(i)
            continue
        if i.halite == 0 and i.ship == None:
            continue
        targets.append(i)
    rewards = np.zeros((len(assign), len(targets)))

    for i,ship in enumerate(assign):
        for j,cell in enumerate(targets):
            rewards[i, j] = get_reward(ship,cell)

    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True) # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        action[assign[r]] = (rewards[r][c],assign[r],targets[c].position)

    #Process actions
    actions = list(action.values())
    actions.sort(reverse=True,key=lambda x : x[0])
    for act in actions:
        act[1].next_action = a_move(act[1],act[2],state[act[1]]['blocked'])
        # Ship convertion
        sPos = act[1].position 
        if state['closestShipyard'][sPos.x][sPos.y] == sPos and state['board'].cells[sPos].shipyard == None:
            act[1].next_action = ShipAction.CONVERT
    return

def spawn_tasks():
    shipyards = state['board'].current_player.shipyards
    shipyards.sort(reverse=True,key=lambda shipyard : state['haliteSpread'][shipyard.position.x][shipyard.position.y])

    for shipyard in shipyards:
        if state['currentHalite'] > 500 and not state['next'][shipyard.cell.position.x][shipyard.cell.position.y]:
            if state['shipValue'] > 500:
                shipyard.next_action = ShipyardAction.SPAWN   
                state['currentHalite'] -= 500
            elif shipyard.cell.ship != None and not shipyard.cell.ship.player_id == state['me']:
                shipyard.next_action = ShipyardAction.SPAWN   
                state['currentHalite'] -= 500
            elif len(state['myShips']) <= 2:
                shipyard.next_action = ShipyardAction.SPAWN   
                state['currentHalite'] -= 500

def convert_tasks():
    global action

    # Add convertion tasks

    rewardMap = shipyard_reward_map() # Best area to build a shipyard
    currentShipyards = state['myShipyards'] # Shipyards "existing"
    targetShipyards = currentShipyards[:]

    t = np.where(rewardMap==np.amax(rewardMap))
    tx,ty = list(zip(t[0], t[1]))[0]

    # Calculate the reward for each cell

    if len(currentShipyards) == 0:
        # Grab the closest ship to the target and build.
        closest  = closest_ship(Point(tx,ty))
        action[closest] = (math.inf,closest,Point(tx,ty))
        targetShipyards.append(state['board'].cells[Point(tx,ty)])
        state['currentHalite'] -= 500
    elif len(state['myShips']) >= len(currentShipyards) * 5 and len(state['myShipyards']) < 4:
        targetShipyards.append(state['board'].cells[Point(tx,ty)])
        state['currentHalite'] -= 500

    state['closestShipyard'] = closest_shipyard(targetShipyards)
