from dependency import *
from navigation import *

# XXX: bulky file, please do something

action = {}  # ship -> (value,ship,target)

def ship_tasks():  # update action
    global action
    cfg = state['configuration']
    board = state['board']
    me = board.current_player
    tasks = {}
    shipsToAssign = []

    # Split attack ships and mine ships
    temp = get_targets()
    state['attackers'] = []
    if len(temp) > 0:
        minerNum = miner_num()
        attackerNum = len(state['myShips']) - minerNum
        for ship in me.ships:
            if ship in action:
                continue
            if attackerNum > 0:
                attackerNum -= 1
                #Uncomment to activate attack
                #state['attackers'].append(ship)

    # All ships rule based
    for ship in me.ships:
        # Flee
        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (INF*2+ship.halite, ship, state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue # continue its current action

        # End-game return
        if board.step > state['configuration']['episodeSteps'] - cfg.size * 2 and ship.halite > 0:
            action[ship] = (ship.halite, ship, state['closestShipyard'][ship.position.x][ship.position.y])
        # End game attack
        if len(state['board'].opponents) > 0 and board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite == 0:
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

        if ship in action or ship in state['attackers']:
            continue

        shipsToAssign.append(ship)

    # Rule based: Attackers
    #print(len(state['myShips']))
    #print(len(state['attackers']))
    attack(state['attackers'])

    # Reward based: Mining + Guarding + Control
    targets = [] # (cell, type)
    for i in board.cells.values():  # Filter targets
        if i.shipyard != None and i.shipyard.player_id == state['me']:
            targets.append((i,'guard'))
            for j in range(min(6,len(state['myShips']))):
                targets.append((i,'cell'))
            continue
        if i.halite < 15 and i.ship == None and i.shipyard == None:
            # Spots not very interesting
            continue
        if i.ship != None and i.ship.player_id != state['me']:
            if i.ship.halite == 0 and state['controlMap'][i.position.x][i.position.y] < 0:
                continue
        targets.append((i,'cell'))
    rewards = np.zeros((len(shipsToAssign), len(targets)))
    for i, ship in enumerate(shipsToAssign):
        for j, target in enumerate(targets):
            rewards[i, j] = get_reward(ship, target)          
    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True)  # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        task = targets[c]
        if task[1] == 'cell':
            action[shipsToAssign[r]] = (rewards[r][c], shipsToAssign[r], targets[c][0].position)
        elif task[1] == 'guard':
            action[shipsToAssign[r]] = (0, shipsToAssign[r], targets[c][0].position)

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
    act[1].next_action = d_move(act[1], act[2], state[act[1]]['blocked'])
    # Ship convertion
    sPos = act[1].position
    if state['closestShipyard'][sPos.x][sPos.y] == sPos and state['board'].cells[sPos].shipyard == None:
        act[1].next_action = ShipAction.CONVERT
    return act[1].next_action

def miner_num():
    
    if state['board'].step < 300:
        if len(state['myShips']) > 25:
            return min(len(state['myShips']),int(state['haliteMean'] / 8 + len(state['myShipyards'])))
        else:
            return min(len(state['myShips']),int(state['haliteMean'] / 4 + len(state['myShipyards'])))
    else:
        return len(state['myShips']) * 0.8
'''
def get_besiege():
    global state
    targets = []
    for x in range(0,21,2):
        for y in range(0,21,2):
            halite = 0
            for i in range(x,x+5):
                for j in range(y,y+5):
                    if 0 < state['enemyShipHalite'][i%21][j%21] <= 5000:
                        halite+=state['enemyShipHalite'][i%21][j%21]+500
            targets.append([x,y,halite])
    targets = targets.sort(key = lambda x: x[2])
    return targets[0]
'''
def get_targets():
    targets = []
    for ship in state['enemyShips']:
        if ship.halite != 0:
            targets.append(ship)
    targets = targets.sort(reverse = True)
    return targets[0]

'''
def greedy_selection(ship):
# Force return
    if ship.halite > 0:
        action[ship] = (INF, ship, state['closestShipyard'][ship.position.x][ship.position.y])            continue
        # Attack
    finalTarget = targets[0]
    v = rule_attack_reward(ship,finalTarget,target_list)
    for target in targets:
        tv = rule_attack_reward(ship,target,target_list)
        if tv > v:
            v = tv
            finalTarget = target
    #target_list.append(finalTarget)
    action[ship] = (1/dist(finalTarget.position,ship.position), ship, finalTarget.position)
'''

def attack(ships):
    global action
    # Select potential targets
    target = get_targets()
    for ship in ships:
        if target.position.x-2<=ship.position.x<=target.position.x+2 and target.position.y-2<=ship.position.y<=target.position.y+2 and ship.halite<target.halite:
            action[ship] = (INF, ship, target.position)


# Greedy selection 
# TODO: Improve this!
def rule_attack_reward(s,t,target_list):
    tPos = t.position 
    sPos = s.position
    d = dist(tPos,sPos)
    res = 1/d
    if t.player == state['killTarget']:
        res = res * 4

    control = state['positiveControlMap'][tPos.x][tPos.y]
    if control > 1 and d < 8:
        # Check if local maxima
        yes = True
        for x in range(-3,4):
            if not yes:
                break
            for y in range(-3,4):
                xx = (tPos.x+x) % 21
                yy = (tPos.y+y) % 21
                if not yes:
                    break
                if state['positiveControlMap'][xx][yy] > control and state['enemyShipHalite'][xx][yy] < 99999 and state['enemyShipHalite'][xx][yy] > 0:
                    yes = False
        if yes:
            res = res * 8
    '''
    for pos in get_adjacent(tPos):
        if state['enemyShipHalite'][pos.x][pos.y] <= s.halite:
            return 0
    '''

    return res


