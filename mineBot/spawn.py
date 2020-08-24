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
    shouldSpawn = spawn()

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



spawnMean = np.array([4.9859e+03, 6.0502e+01, 2.5001e+01, 1.9415e+02, 2.8910e+01, 6.1503e-01])
spawnStd = np.array([8.5868e+03, 1.5326e+01, 1.0737e+01, 1.1549e+02, 1.1789e+01, 4.8660e-01])

W1 = np.array([[-1.5224804e+00,2.4725301E-03,-8.7220293e-01,-1.0598649e+00,
   9.9166840e-01,1.8315561e+00],
 [-4.8011017e-01,-6.7499268e-01 ,3.5633636e-01,-1.7301080e+00,
   2.0809724e+00,-8.9656311e-01],
 [-1.1370039e+00,-2.0581658e-01,-2.6484251e+00,-1.5524467e+00,
   3.5835698e+00,-1.7890360e+00],
 [-1.7479208e-01 ,1.9892944e-01, 1.4682317e-01 , 1.1079860e+00,
   1.4466201e-01 , 1.9152831e+00]])
b1 = np.array([1.177493, 0.5530099, 0.1025302, 2.165062 ])

W2 = np.array([[ 0.22407304 ,-0.32596582 ,-0.31062314 ,-0.17025752],
 [-3.6107817 ,  1.9571906 , -0.04028177, -4.0320687 ],
 [ 4.130036  , -1.2309656,  -0.52751654,  1.5594524 ],
 [-0.33959138, -0.0332855 , -0.26249635, -0.35909724]])
b2 = np.array([-0.40560475 ,-0.00167005 , 0.7714385 , -0.19049597])

W3 = np.array([[ 0.4247551 ,  5.073255   ,-4.3405128  , 0.00574893]])
b3 = np.array([-0.2889765])

def ship_value():
    if len(state['myShips']) >= 60:
        return 0
    res = state['haliteMean'] * 0.25 * (state['configuration']['episodeSteps']- 30 - state['board'].step) * weights[4][0]
    res += (len(state['ships']) - len(state['myShips'])) ** 1.5 * weights[4][1]
    res += len(state['myShips'])  ** 1.5 * weights[4][2]
    return res 
