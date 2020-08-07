def miner_num():
    if state['board'].step < 300:
        return min(len(state['myShips']),int(state['haliteMean'] / 2 + len(state['myShipyards'])))
    else:
        return len(state['myShips'])

def attack(ships):
    global action

    # Select potential targets
    targets = []
    for ship in state['enemyShips']:
        if ship.halite != 0:
            targets.append(ship)
    # Execute
    for ship in ships:
        # Force return
        if ship.halite > 0:
            action[ship] = (INF, ship, state['closestShipyard'][ship.position.x][ship.position.y])
            continue
        # Attack
        finalTarget = targets[0]
        v = rule_attack_reward(ship,finalTarget)
        for target in targets:
            tv = rule_attack_reward(ship,target)
            if tv > v:
                v = tv
                finalTarget = target
        action[ship] = (0, ship, finalTarget.position)

# Greedy selection 
# TODO: Improve this!
def rule_attack_reward(s,t):
    tPos = t.position 
    sPos = s.position
    
    res = 1/dist(tPos,sPos)
    if t.player == state['killTarget']:
        res = res * 2

    for pos in get_adjacent(tPos):
        if state['enemyShipHalite'][pos.x][pos.y] <= s.halite:
            return 0

    return res
