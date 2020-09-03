# The final function


@board_agent
def agent(board):

    print("Turn =", board.step + 1)
    # Init
    if board.step == 0:
        init(board)
    # Encode
    encode(board)

    # Convert
    convert_tasks()

    # Ship
    ship_tasks()

    # Spawn
    spawn_tasks()

    