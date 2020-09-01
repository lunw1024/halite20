def spawn_tasks():
    shipyards = state["board"].current_player.shipyards
    shipyards.sort(reverse=True, key=lambda shipyard: state["haliteSpread"][shipyard.position.x][shipyard.position.y])

    for shipyard in shipyards:
        if state["currentHalite"] >= 500 and not state["next"][shipyard.cell.position.x][shipyard.cell.position.y]:
            step = state["board"].step
            enemyShipCnt = [len(p.ships) for p in state['board'].opponents]
            if step < state["configuration"]["episodeSteps"] // 5 * 4 and len(state["myShips"]) <= max(enemyShipCnt):
                shipyard.next_action = ShipyardAction.SPAWN
            elif len(state["myShipyards"]) == 1:
                for pos in get_adjacent(shipyard.position):  # enemy nearby -> reinforce
                    cell = state["board"].cells[pos]
                    if cell.ship is not None and cell.ship.player_id != state["me"]:
                        shipyard.next_action = ShipyardAction.SPAWN
                        state["currentHalite"] -= 500
