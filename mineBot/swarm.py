INFLUENCE_RANGE = 4
def select_swarm_target():
    board = state['board']
    N = state['configuration'].size

    # We want to target player with fewer shipyards
    shipyardScores = []
    for shipyard in state['shipyards']:
        if shipyard.player == board.current_player:
            continue
        sPos = shipyard.position
        v = 0
        for x in range(sPos.x-INFLUENCE_RANGE,sPos.x+INFLUENCE_RANGE+1):
            for y in range(sPos.y-INFLUENCE_RANGE,sPos.y+INFLUENCE_RANGE+1):
                xx = (x + N) % N
                yy = (y + N) % N
                cell = board.cells[Point(xx,yy)]
                v += cell.halite
                
                '''
                if cell.ship != None and cell.ship.player == shipyard.player and cell.ship.halite > 0:
                    v += 20
                elif cell.ship != None and cell.ship.player == shipyard.player and cell.ship.halite == 0:
                    v -= 10
                '''
                
        if not state['swarm'] is None and shipyard.position == state['swarm']:
            v *= 2
        v = v / max(1,len(shipyard.player.shipyards))
        shipyardScores.append((v,shipyard))
    if len(shipyardScores) > 0:
        state['swarm'] = max(shipyardScores,key=lambda x: x[0])[1].position

def get_swarm_attack_targets():
    targets = []
    for ship in state['enemyShips']:
        if ship.halite != 0 and dist(ship.position,state['swarm']) <= 5:
            targets.append(ship)
    return targets

def swarm(ships):
    global action
    board = state['board']
    select_swarm_target()
    N = state['configuration'].size


    if len(ships) == 0:
        return 

    shipsToAssign = []
    for ship in ships:
        if ship in action:
            continue
        shipsToAssign.append(ship)

    # Get the attack base
    target = state['swarm']
    if target is None:
        for ship in shipsToAssign:
            state['miners'].append(ship)
        return 
    tPos = target
    
    # Get our FOB

    FOB = state['closestShipyard'][tPos.x][tPos.y]
    if dist(FOB,tPos) > INFLUENCE_RANGE:
        FOB = None 
    if FOB is None:
        potentialFOB = []
        for x in range(tPos.x-INFLUENCE_RANGE,tPos.x+INFLUENCE_RANGE+1):
            for y in range(tPos.y-INFLUENCE_RANGE,tPos.y+INFLUENCE_RANGE+1):
                xx = (x + N) % N
                yy = (y + N) % N
                t = Point(xx,yy)
                if dist(t,tPos) <= 1 or dist(t,tPos) > INFLUENCE_RANGE:
                    continue
                v = state['haliteSpread'][xx][yy] / dist(t,tPos)
                if board.cells[t].shipyard != None:
                    continue
                potentialFOB.append((v,t))
        FOB = max(potentialFOB,key=lambda x: x[0])[1]
    
    # Build our FOB if it doesn't exist
    if board.cells[FOB].shipyard == None and state['currentHalite'] > 500:       
        state['targetShipyards'].append(board.cells[FOB])
        state['currentHalite'] -= 500
        state['closestShipyard'] = closest_shipyard(state['targetShipyards'])
        
        for ship in shipsToAssign:
            action[ship] = (0,ship,FOB)

    targets = get_swarm_attack_targets()
    for ship in shipsToAssign:
        if ship in action:
            continue
        if len(targets) > 0:
            if ship.halite > 0:
                action[ship] = (INF, ship, state['closestShipyard'][ship.position.x][ship.position.y])
                continue
            finalTarget = targets[0]
            v = rule_swarm_reward(ship,finalTarget)
            for target in targets:
                tv = rule_swarm_reward(ship,target)
                if tv > v:
                    v = tv
                    finalTarget = target
        else:
            state['miners'].append(ship)
            continue

        action[ship] = (1/dist(finalTarget.position,ship.position), ship, finalTarget.position)

def rule_swarm_reward(s,t):

    tPos = t.position 
    sPos = s.position
    d = dist(tPos,sPos)
    res = 1/d
    if t.player == state['killTarget']:
        res = res * 4

    control = state['positiveControlMap'][tPos.x][tPos.y]
    if state['trapped'][t.player_id][tPos.x][tPos.y] and d <= 6:
        res = res * 10
    return res




    

    


         

    
        





