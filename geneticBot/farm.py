farms = [] # list of cells to farm

def farm_tasks():
    control_farm()
    if len(farms) < 4 * len(state['myShipyards']):
        build_farm()
    # Create patrols

def build_farm():
    global farms
    maxCell = None
    v = 0

    for cell in state['board'].cells.values():
        if cell.position in farms:
            continue
        a = farm_value(cell)
        if a > v:
            maxCell = cell
            v = a

    if maxCell != None:
        farms.append(maxCell.position)

def control_farm():
    global farms
    for i,farm in enumerate(farms[:]):
        if state['board'].cells[farm].halite < state['haliteMean'] / 1.5 or dist(farm,state['closestShipyard'][farm.x][farm.y]) > 8:
            # Not worth it
            farms.remove(farm)

def farm_value(cell):
    cPos = cell.position
    if len(state['myShipyards']) == 0 or cell.halite == 0:
        return 0

    closest = state['closestShipyard'][cPos.x][cPos.y]
    if dist(closest,cPos) <= 1 or dist(closest,cPos) > 4:
        return 0

    return (cell.halite**0.5) / dist(closest,cPos) ** 2