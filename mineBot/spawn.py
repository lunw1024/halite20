def spawn():
    # Ship value: 
    '''
    if state['shipValue'] >= 500: 
        return True
    else:
        return False
    '''
    # 抄袭
    bank = state['currentHalite']
    haliteMean = state['haliteMean']
    step = state['board'].step
    shipCnt = len(state['myShips'])
    totalShipCnt = len(state['ships'])
    #isBlocked = state['next'][shipyard.cell.position.x][shipyard.cell.position.y]
    isBlocked = 0 #In theory never blocked, as already checked

    if shipCnt >= 60 or step > 330:
        return False

    inArr = (np.array([bank, totalShipCnt, shipCnt, step, haliteMean, isBlocked]) - spawnMean) / spawnStd
    res = W1 @ inArr + b1
    res = np.maximum(res, 0)
    res = W2 @ res + b2
    res = np.maximum(res, 0)
    res = W3 @ res + b3
    #print(res)
    if res > 0:
        return True
    else:
        return False

def spawn_tasks():
    shipyards = state['board'].current_player.shipyards
    shipyards.sort(reverse=True, key=lambda shipyard: state['haliteSpread'][shipyard.position.x][shipyard.position.y])
    shouldSpawn = state['spawn']

    for shipyard in shipyards:
        if state['currentHalite'] >= 500 and not state['next'][shipyard.cell.position.x][shipyard.cell.position.y]:
            if shouldSpawn:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShips']) < 1 and shipyard == shipyards[0]:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShipyards']) == 1:
                for pos in get_adjacent(shipyard.position):
                    cell = state['board'].cells[pos]
                    if cell.ship != None and cell.ship.player_id != state['me']:
                        shipyard.next_action = ShipyardAction.SPAWN
                        state['currentHalite'] -= 500
                        return



spawnMean = np.array([7.6773e+03, 5.7181e+01, 1.0523e+01, 1.9315e+02, 2.0789e+01, 3.6286e-01])
spawnStd = np.array([5.0098e+03, 1.4527e+01, 4.0529e+00, 1.1240e+02, 1.5751e+01, 4.8083e-01])

W1 = np.array([[ 0.4283,  0.2175,  0.0909,  0.0074, -0.2957, -0.3837],
        [-0.4558,  0.0032, -0.1289, -0.1063,  0.1216,  0.9327],
        [ 0.5735, -0.2480, -0.2565,  1.1325,  0.1377,  0.3335],
        [ 0.5214, -0.2850, -0.1977, -0.3675,  0.5467, -0.8464]])
b1 = np.array([0.4985, 0.1115, 0.0743, 0.5155])

W2 = np.array([[-1.2372, -1.4252, -1.0081,  0.8310],
        [-1.5292, -1.5950, -0.8577,  0.7841],
        [-1.0459, -1.4518, -1.0939,  0.7897],
        [-0.3721, -0.6974, -0.5525, -0.4736]])
b2 = np.array([0.2956, 0.4722, 0.2449, 0.0123])

W3 = np.array([[1.0935, 1.1482, 1.0861, 0.2745]])
b3 = np.array([-4.6521])
