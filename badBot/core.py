# Core strategy

def ship_tasks(): # return updated tasks
    cfg = state['configuration']
    board = state['board']
    me = board.current_player
    tasksOld = tasks.copy()
    tasks.clear()
    
    # calculate rewards
    targets = list(board.cells.values())
    rewards = np.zeros((len(me.ships), len(targets))) # reward matrix for optimization
    for i, ship in enumerate(me.ships):
        for j, cell in enumerate(targets): # enumerate targets TODO: enemy ships/shipyards
            rewards[i, j] = getReward(ship,cell)
    
    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True) # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        tasks[me.ships[r]] = targets[c]

    for ship, target in tasks.items():
        candidates = directions_to(ship.position, target.position)
        for candidate in candidates:
            nextCell = board.next()[dry_move(ship.position, candidate)]
            if nextCell.ship is None or not nextCell.ship.player.is_current_player: # not ally ship
                ship.next_action = candidate
                break

    return

def spawn_tasks():
    # Add shipyard tasks
    pass

def convert_tasks():
    # Add convertion tasks
    pass
    