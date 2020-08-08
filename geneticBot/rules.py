def miner_num():
    
    if state['board'].step < 300:
        if len(state['myShips']) > 25:
            return min(len(state['myShips']),int(state['haliteMean'] / 8 + len(state['myShipyards'])))
        else:
            return min(len(state['myShips']),int(state['haliteMean'] / 4 + len(state['myShipyards'])))
    else:
        return len(state['myShips']) * 0.8

def attack(ships):
    global action

    # Select potential targets
    targets = []
    for ship in state['enemyShips']:
        if ship.halite != 0:
            targets.append(ship)
    # Execute
    target_list = []
    for ship in ships:
        # Force return
        if ship.halite > 0:
            action[ship] = (INF, ship, state['closestShipyard'][ship.position.x][ship.position.y])
            continue
        # Attack
        finalTarget = targets[0]
        v = rule_attack_reward(ship,finalTarget,target_list)
        for target in targets:
            tv = rule_attack_reward(ship,target,target_list)
            if tv > v:
                v = tv
                finalTarget = target
        target_list.append(finalTarget)
        action[ship] = (0, ship, finalTarget.position)

# Greedy selection 
# TODO: Improve this!
def rule_attack_reward(s,t,target_list):
    tPos = t.position 
    sPos = s.position
    #colaborate
    colaborators = target_list.count(t)
    res = 1/dist(tPos,sPos)
    if t.player == state['killTarget']:
        res = res * 2
    
    res = res * t.halite
    res = res * colaborators
    
    '''
    for pos in get_adjacent(tPos):
        if state['enemyShipHalite'][pos.x][pos.y] <= s.halite:
            return 0
    '''

    return res
