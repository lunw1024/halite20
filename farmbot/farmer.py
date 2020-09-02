# XXX
SCHEMA = [((-1, 2), (0, 3)), ((1, 3), (2, 2)),
          ((-1, -3), (0, -3)), ((1, -2), (2, -2)),
          ((2, 1), (3, 0)), ((3, 1), (-2, -2)),
          ((-2, 2), (-3, 0)), ((-2, -1), (-3, 1))]


def wall_schema(): 
    # design wall structure according to shipyard pos
    # out[i, j] > 0 means (i, j) should be wall, the 
    # value indicates the pair (to swap non-stop)
    # TODO: remove connected walls
    cfg = state["configuration"]
    board = state["board"]
    me = board.current_player
    N = cfg.size
    out = np.zeros((N, N), dtype=int)
    for shipyard in me.shipyards:
        x, y = shipyard.position
        for i, ((dx1, dy1), (dx2, dy2)) in enumerate(SCHEMA):
            out[(x + dx1) % N, (y + dy1) % N] = i + 1
            out[(x + dx2) % N, (y + dy2) % N] = i + 1
    return out

def farm(ships):
    pass