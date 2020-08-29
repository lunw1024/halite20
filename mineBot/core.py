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

    # Split attack ships and mine ships

    #=====================#
    # Currently only miners. Structure below is for reference
    temp = get_attack_targets()

    state['attackers'] = []
    state['swarmers'] = []
    state['miners'] = []
    minerNum = miner_num()
    for ship in state['myShips']:
        if minerNum > 0:
            minerNum -= 1
            state['miners'].append(ship)
        else:
            #state['swarmers'].append(ship)
            state['miners'].append(ship)
    #=====================#

    # All ships rule based
    for ship in me.ships:

        if ship in action:
            continue 

        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (INF*2+state[ship]['danger'][ship.position.x][ship.position.y], ship, state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue # continue its current action

        # End-game return
        if board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite > 0:
            action[ship] = (ship.halite, ship, state['closestShipyard'][ship.position.x][ship.position.y])
        # End game attack
        if len(state['board'].opponents) > 0 and board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite == 0:
            #print(ship.position)
            if len(state['myShipyards']) > 0 and ship == closest_thing(state['myShipyards'][0].position,state['myShips']):
                action[ship] = (0,ship,state['myShipyards'][0].position)
                continue
            killTarget = state['killTarget']
            if len(killTarget.shipyards) > 0:
                target = closest_thing(ship.position,killTarget.shipyards)
                action[ship] = (ship.halite, ship, target.position)
            elif len(killTarget.ships) > 0:
                target = closest_thing(ship.position,killTarget.ships)
                action[ship] = (ship.halite, ship, target.position)

    #attack(state['attackers'])
    swarm(state['swarmers'])
    mine(state['miners'])

    # Reward based: Mining + Guarding + Control
            
    # Process actions
    actions = list(action.values())
    actions.sort(reverse=True, key=lambda x: x[0])
    for act in actions:
        process_action(act)

def miner_num():
    if len(state['myShips']) < 18:
        return len(state['myShips'])
    if state['board'].step < 280:
        if len(state['myShips']) > 25:
            return min(len(state['myShips']),int(state['haliteMean'] / 4 + len(state['myShipyards'])))
        else:
            return min(len(state['myShips']),int(state['haliteMean'] / 2 + len(state['myShipyards'])))
    elif state['board'].step > 370:
        return len(state['myShips'])
    else:
        return len(state['myShips']) * 0.8

def process_action(act):
    global action
    if action[act[1]] == True:
        return act[1].next_action
    action[act[1]] = True
    # Processing
    act[1].next_action = d_move(act[1], act[2], state[act[1]]['blocked'])
    # Ship convertion
    sPos = act[1].position
    if state['closestShipyard'][sPos.x][sPos.y] == sPos and state['board'].cells[sPos].shipyard == None:
        act[1].next_action = ShipAction.CONVERT
        state['next'][sPos.x][sPos.y] = 1
    return act[1].next_action
