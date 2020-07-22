# The final function


@board_agent
def agent(board):

    print("Turn =",board.step+1)
    # Init
    if board.step == 0:
        init(board)

    # Update
    update(board)

    # Convert
    convert_tasks()

    # Ship
    ship_tasks()

    print(state['shipValue'])

    # Spawn
    spawn_tasks()