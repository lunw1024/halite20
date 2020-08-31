def farm_tasks():
    build_farm()
    control_farm()
    # Create patrols

def build_farm():
    global farms
    for cell in state['board'].cells.values():
        if dist(cell.position,state['closestShipyard'][cell.position.x][cell.position.y]) == 1:
            if cell.position in farms:
                continue
            farms.append(cell.position)

def control_farm():
    global farms
    for i,farm in enumerate(farms[:]):
        if dist(farm,state['closestShipyard'][farm.x][farm.y]) > 1:
            # Not worth it
            farms.remove(farm)
