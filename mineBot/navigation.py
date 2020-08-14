# Direction from point s to point t
def direction_to(s: Point, t: Point) -> ShipAction:
    candidate = directions_to(s, t)
    return random.choice(candidate) if len(candidate) > 0 else None

# Distance from point a to b
def dist(a: Point, b: Point) -> int:
    N = state['configuration'].size
    return min(abs(a.x - b.x), N - abs(a.x - b.x)) + min(abs(a.y - b.y), N - abs(a.y - b.y))

# Returns list of possible directions
def directions_to(s: Point, t: Point) -> ShipAction:
    N = state['configuration'].size
    candidates = [] # [N/S, E/W]
    if s.x-t.x != 0:
        candidates.append(ShipAction.WEST if (s.x-t.x) % N < (t.x-s.x) % N else ShipAction.EAST)
    if s.y-t.y != 0:
        candidates.append(ShipAction.SOUTH if (s.y-t.y) % N < (t.y-s.y) % N else ShipAction.NORTH)
    return candidates

# Deserialize an integer which represents a point
def unpack(n) -> Point:
    N = state['configuration'].size
    return Point(n // N, n % N)

# A default direction to target
def direction_to(s: Point, t: Point) -> ShipAction:
    candidate = directions_to(s, t)
    return random.choice(candidate) if len(candidate) > 0 else None

# Returns the "next" point of a ship at point s with shipAction d
def dry_move(s: Point, d: ShipAction) -> Point:
    N = state['configuration'].size
    if d == ShipAction.NORTH:
        return s.translate(Point(0, 1),N)
    elif d == ShipAction.SOUTH:
        return s.translate(Point(0, -1),N)
    elif d == ShipAction.EAST:
        return s.translate(Point(1, 0),N)
    elif d == ShipAction.WEST:
        return s.translate(Point(-1, 0),N)
    else:
        return s
    
# Returns opposite direction
def opp_direction(d: ShipAction):
    if d == ShipAction.NORTH:
        return ShipAction.SOUTH
    if d == ShipAction.SOUTH:
        return ShipAction.NORTH
    if d == ShipAction.WEST:
        return ShipAction.EAST
    if d == ShipAction.EAST:
        return ShipAction.WEST
    return None

# Returns list of len 4 of adjacent points to a point
def get_adjacent(point):
    N = state['configuration'].size
    res = []
    for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
        res.append(point.translate(Point(offX,offY),N))
    return res
    
def safe_naive(s,t,blocked):
    for direction in directions_to(s.position,t):
        target = dry_move(s.position,direction)
        if not blocked[target.x][target.y]:
            return direction
    return None

def move_cost(s : Ship, t : Point):
    navigationWeights = weights[6]
    cost = state[s]['danger'][t.x][t.y] * navigationWeights[1]
    for pos in get_adjacent(t):
        if state['enemyShipHalite'][pos.x][pos.y] == s.halite:
            cost += navigationWeights[0]
    return cost

# Dijkstra's movement
def d_move(s : Ship, t : Point, inBlocked):

    nextMap = state['next']
    sPos = s.position
    blocked = inBlocked + nextMap
    # Check if we are trying to attack
    if state['board'].cells[t].ship != None:
        target = state['board'].cells[t].ship
        if target.player_id != state['me'] and target.halite == s.halite:
            blocked[t.x][t.y] -= 1
    elif state['board'].cells[t].shipyard != None and state['board'].cells[t].shipyard.player_id != state['me']:
        blocked[t.x][t.y] -= 1
    # Don't ram stuff thats not the target. Unless we have an excess of ships. Or we are trying to murder a team.
    if state['board'].step < state['configuration']['episodeSteps'] - state['configuration'].size * 1.5:
        temp = np.where(state['enemyShipHalite'] == s.halite, 1, 0)
        blocked += temp
        
        adjacent = np.zeros(blocked.shape)
        if state['haliteMean'] <= 25 or s in state['attackers']:
            adjacent += np.roll(temp,1,axis=0)
            adjacent += np.roll(temp,1,axis=1)
            adjacent += np.roll(temp,-1,axis=0)
            adjacent += np.roll(temp,-1,axis=1)
            adjacent -= state['allyShipyard'] * 5
            adjacent = np.where(adjacent>0,1,0)
            blocked+=adjacent
        

    blocked = np.where(blocked>0,1,0)

    desired = None

    #Stay still
    if sPos == t or nextMap[t.x][t.y]:

        #Someone with higher priority needs position, must move. Or being attacked.
        if blocked[t.x][t.y]:
            for processPoint in get_adjacent(sPos):
                if not blocked[processPoint.x][processPoint.y]:
                    nextMap[processPoint.x][processPoint.y] = 1
                    desired = direction_to(sPos,processPoint)
                    t = processPoint
            if desired == None:
                target = micro_run(s)
                t = dry_move(sPos,target)
                desired = target
        else:
            t = sPos
            desired = None
    else:
        #Dijkstra
        pred = {}
        calcDist = {}
        pq = PriorityQueue()
        pqMap = {}

        pqMap[dist(sPos,t)] = [sPos]
        pq.put(dist(sPos,t))
        pred[sPos] = sPos
        calcDist[sPos] = dist(sPos,t)

            # Main

        while not pq.empty():
            if t in calcDist:
                break
            currentPoint = pqMap.get(pq.get()).pop()
            for processPoint in get_adjacent(currentPoint):
                if blocked[processPoint.x][processPoint.y] or processPoint in calcDist: 
                    continue
                calcDist[processPoint] = calcDist[currentPoint] + 1 + move_cost(s,processPoint)
                priority = calcDist[processPoint]
                pqMap[priority] = pqMap.get(priority,[])
                pqMap[priority].append(processPoint)
                pq.put(priority)
                pred[processPoint] = currentPoint

        if not t in pred:

            # Can go in general direction
            res = safe_naive(s,t,blocked)
            if res != None:
                t = dry_move(s.position,res)
                desired = res
            else:
                #Random move
                for processPoint in get_adjacent(sPos):
                    if not blocked[processPoint.x][processPoint.y]:
                        nextMap[processPoint.x][processPoint.y] = 1
                        t = processPoint
                        desired = direction_to(sPos,processPoint)
                
                # Run
                if desired == None and blocked[sPos.x][sPos.y]:
                    target = micro_run(s)
                    t = dry_move(sPos,target)
                    desired = target
                elif not blocked[sPos.x][sPos.y]:
                    t = sPos
                    desired = None        
        else:
            # Path reconstruction
            while pred[t] != sPos:
                t = pred[t]

            desired = direction_to(sPos,t)

    # Reduce collisions
    if desired != None and state['board'].cells[t].ship != None and state['board'].cells[t].ship.player_id == state['me']:
        target = state['board'].cells[t].ship
        s.next_action = desired
        if action[target] != True:
            nextMap[t.x][t.y] = 1
            result = process_action(action[target])
            # Going there will kill it
            if result == None or result == ShipAction.CONVERT:
                desired = d_move(s,t,inBlocked)
                t = dry_move(sPos,desired)
    nextMap[t.x][t.y] = 1
    return desired

# Ship might die, RUN!
def micro_run(s):
    sPos = s.position
    nextMap = state['next']

    if state[s]['blocked'][sPos.x][sPos.y]:
        if s.halite > 400:
            return ShipAction.CONVERT
        score = [0,0,0,0]

        # Preprocess
        directAttackers = 0
        for i,pos in enumerate(get_adjacent(sPos)):
            if state['enemyShipHalite'][pos.x][pos.y] < s.halite:
                directAttackers += 1

        # Calculate score
        for i,pos in enumerate(get_adjacent(sPos)):
            score[i] = 0
            for j,tPos in enumerate(get_adjacent(sPos)):
                if state['enemyShipHalite'][tPos.x][tPos.y] < s.halite:
                    score[i] -= 0.5
            if state['enemyShipHalite'][pos.x][pos.y] < s.halite:
                score[i] -= 0.5 + 1/directAttackers
            score[i] -= state['negativeControlMap'][pos.x][pos.y] * 0.01
        # Select best position
        i, maximum = 0,0 
        for j, thing in enumerate(score):
            if thing > maximum:
                i = j
                maximum = thing
        return direction_to(sPos,get_adjacent(sPos)[i])
    else:
        return None





