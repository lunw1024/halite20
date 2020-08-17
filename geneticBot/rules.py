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
