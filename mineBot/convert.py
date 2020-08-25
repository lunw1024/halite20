def convert_tasks():
    global action

    # Add convertion tasks

    currentShipyards = state['myShipyards']  # Shipyards "existing"
    targetShipyards = currentShipyards[:]

    # Maximum cell
    v = shipyard_value(state['board'].cells[Point(0,0)])
    t = state['board'].cells[Point(0,0)]
    for cell in state['board'].cells.values():
        a = shipyard_value(cell)
        if v < a:
            v = a
            t = cell
    tx, ty = t.position.x,t.position.y
    # Calculate the reward for each cell
    if state['board'].step == 0:
        # Build immediately
        targetShipyards.append(state['board'].cells[state['myShips'][0].position])
        action[state['myShips'][0]] = (math.inf, state['myShips'][0], state['myShips'][0].position)
        state['currentHalite'] -= 500
    elif len(currentShipyards) == 0:
        # Grab the closest possible ship to the target and build.
        possibleShips = []
        for ship in state['myShips']:
            if ship.halite + state['currentHalite'] >= 500:
                possibleShips.append(ship)
        closest = closest_thing(Point(tx, ty),possibleShips)
        if closest != None:
            action[closest] = (math.inf, closest, Point(tx, ty))
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500
    elif v > 500 and v > state['shipValue']:
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500
    
    state['targetShipyards'] = targetShipyards
    state['closestShipyard'] = closest_shipyard(targetShipyards)

def shipyard_value(cell):
    # Features
    shipyardWeights = weights[0]
    cPos = cell.position

    if state['board'].step > 310:
        return 0

    nearestShipyard = closest_thing(cPos,state['shipyards'])
    nearestShipyardDistance = 1
    if nearestShipyard != None:
        nearestShipyardDistance = dist(nearestShipyard.position,cPos)
    negativeControl = min(0,state['controlMap'][cPos.x][cPos.y])
    if len(state['myShips']) > 0:
        negativeControl = max(negativeControl-0.5 ** dist(closest_thing(cPos,state['myShips']).position,cPos),state['negativeControlMap'][cPos.x][cPos.y])
    haliteSpread = state['haliteSpread'][cPos.x][cPos.y] - state['haliteMap'][cPos.x][cPos.y]
    shipShipyardRatio = len(state['myShips']) / max(1,len(state['myShipyards']))

    # Hard limit on range and halite spread
    if nearestShipyardDistance <= 5 or haliteSpread <= 200:
        return 0

    # Base halite multiplier
    res = haliteSpread * shipyardWeights[0]

    # Negative control
    res += negativeControl * shipyardWeights[1]

    # Nearest shipyard
    res = res * nearestShipyardDistance ** shipyardWeights[2]

    # Ship shipyard ratio multiplier
    res = res * shipShipyardRatio ** shipyardWeights[3]

    # Final multiplier and bias
    res = res * shipyardWeights[4] + shipyardWeights[5]

    return res