def get_attack_targets():
    targets = []
    for ship in state['enemyShips']:
        if ship.halite != 0:
            targets.append(ship)
    return targets

def attack(ships):
    global action

    # Select potential targets
    targets = get_attack_targets()
    # Greedy selection
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
        action[ship] = (1/dist(finalTarget.position,ship.position), ship, finalTarget.position)

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
            res = res * 6
    
    if state['trapped'][t.player_id][tPos.x][tPos.y] and d <= 6:
        res = res * 10

    '''
    for pos in get_adjacent(tPos):
        if state['enemyShipHalite'][pos.x][pos.y] <= s.halite:
            return 0
    '''

    return res


###################
# target based attack system
###################

'''
def target_based_attack():
    # actions[ship] = (priority: int, ship: Ship, target: Point)
    params = weights[7] # <- np.array
    # target selection
    targets = "all enemy ships with cargo > 0"
    sorted(targets, key="cargo")

    # assignment
    for target in targets:
        actions["all ally ships with cargo < target.cargo" in area5x5(target)] = ("priority", "ship", "target.pos")
'''
