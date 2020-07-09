# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Execution environment

# %%


# %%
print("Import started")
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import random
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from queue import PriorityQueue
print("Import ended")

# %% [markdown]
# # Test Environment

# %%
environment = make("halite", configuration={"size": 21, "startingHalite": 25000}, debug=True)
agent_count = 4
environment.reset(agent_count)
state = environment.state[0]
board = Board(state.observation, environment.configuration)

# %% [markdown]
# # Framework
# 
# ## Static
# Static
# 
# ## Navigation
# Contains helper functions related to *Points* and *Movement*
# 
# #### State variables
# 
# self.next: Numpy array of (SIZE,SIZE) boolean encoded ally unit position on next turn.
# 
# #### Methods
# 
# safeMoveTo: 
# A* "safe" movement
# 
# dist: 
# distance between two Points
# 
# directionTo:
# returns ShipAction. From start to end
# 
# ## Calculator
# Encodes *Board* to numpy array and runs most computationally intensive calculations and heuristics.
# 
# #### Methods
# 
# Update: Runs every turn. A pipeline for all calculations.
# Encode: encodes a board into numpy arrays:
# 
# #### States
# 
# shipMap,shipyardMap: 
# 4D tensor, each dimension a matrix boolean encoding ship/shipyards of a single player (the dimension)
# 
# haliteMap: 
# Matrix of haliteMap
# 
# enemyShipHalite: 
# Matrix of enemyShips, encoded by amount of Halite. Used to threshold.
# 
# ally: 
# My ships.
# 
# controlMap: 
# Heuristic of map control and domination.
# 
# 
# 

# %%
# Static
nav, calc = None, None

class Navigation:

    # Helper
    def __init__(self, board: Board):
        self.CFG = board.configuration

    def dist(self, a: Point, b: Point) -> int:
        return min(abs(a.x - b.x), self.CFG.size - abs(a.x - b.x)) + min(abs(a.y - b.y), self.CFG.size - abs(a.y - b.y))

    def directionTo(self, s: Point, t: Point) -> ShipAction:
        candidate = []  # [N/S, E/W]
        if s.x - t.x != 0:
            candidate.append(ShipAction.WEST if (s.x - t.x) % self.CFG.size < (t.x - s.x) % self.CFG.size else ShipAction.EAST)
        if s.y - t.y != 0:
            candidate.append(ShipAction.SOUTH if (s.y - t.y) % self.CFG.size < (t.y - s.y) % self.CFG.size else ShipAction.NORTH)
        return random.choice(candidate) if len(candidate) > 0 else None

    #Converts a serial representation of x and y
    def unpack(self, n):
        return Point(n // self.CFG.size, n % self.CFG.size)

    #Returns an iterable of adjacent points
    def getAdjacent(self,point):
        res = []
        for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
            res.append(point.translate(Point(offX,offY),self.CFG.size))
        return res

    # Navigation
    def update(self):
        self.next = np.zeros((self.CFG.size,self.CFG.size))

    def safeMoveTo(self, s : Ship, t : Point): #A* Movement. Suggested move by priority.

        sPos = s.position
        blocked = calc.shipMaps[s]['blocked'] + self.next
        blocked = np.where(blocked>0,1,0)

        #print("=====")
        #print(sPos)
        #print(blocked)

        #1. Obstacle Calculation

            #Obstacle are "walls" on the nav graph. Consist of the points of
                #Enemy ships with less halite (threshold => enemy block)
                #Enemy shipyards 
                #Position of friendly on next turn

        #2. Navigation

            #A* 

                #sPos: start position
                #pred: predecessor of a node. (Which point was relaxed to find next point)
                #dist: distance from sPos to point
                #pqMap: maps distances in priority queue to process points 
                #t: initally target point. During reconstruction, becomes "next" point in A* path
                
                
                #algorithm: starts from sPos, put in priority queue.
                #While priority queue is not empty and target is not found, relax next node in queue.
                #Add adjacent (processPoints) to pq.

                #Check if t is reachable (pred not None)
                #If it is, loop back pred until reached sPos to find path.
                #Else, move randomly.
                

            #Swapping
                #If bot wishes to stay still but cannot (self.next turn ally boat moves in)
                #Move randomly
                #This means that if the bot has a goal, it will move toward the goal. This includes friendly
                #As obstacles are calculated through self.next.
                #Because movement is sorted in priority, higher priority ships will never get blocked 
                #By lower priority.


        #TODO: Improve obstacle calculation

        #Stay still
        if sPos == t:
            #Someone with higher priority needs position, must move. Or being attacked.
            if blocked[t.x][t.y]:
                for processPoint in self.getAdjacent(sPos):
                    if not blocked[processPoint.x][processPoint.y]:
                        self.next[processPoint.x][processPoint.y] = 1
                        return self.directionTo(sPos,processPoint)
                self.next[sPos.x][sPos.y] = 1
                return None
            else:
                self.next[sPos.x][sPos.y] = 1
                return None

        #A*
        pred = {}
        dist = {}
        pq = PriorityQueue()
        pqMap = {}

        pqMap[self.dist(sPos,t)] = [sPos]
        pq.put(self.dist(sPos,t))
        pred[sPos] = sPos
        dist[sPos] = self.dist(sPos,t)

            # Main

        while not pq.empty():
            if t in dist:
                break
            currentPoint = pqMap.get(pq.get()).pop()
            for processPoint in self.getAdjacent(currentPoint):
                if blocked[processPoint.x][processPoint.y] or processPoint in dist: 
                    continue
                dist[processPoint] = dist[currentPoint] + 1
                priority =  dist[processPoint] + self.dist(processPoint,t)
                pqMap[priority] = pqMap.get(priority,[])
                pqMap[priority].append(processPoint)
                pq.put(priority)
                pred[processPoint] = currentPoint
        
        #TODO: Catch this exception. Or make sure this never happens. Don't just move randomly.
        if not t in pred:
            #Random move
            for processPoint in self.getAdjacent(sPos):
                if not blocked[processPoint.x][processPoint.y]:
                    self.next[processPoint.x][processPoint.y] = 1
                    return self.directionTo(sPos,processPoint)
            self.next[sPos.x][sPos.y] = 1
            return None

            # Path reconstruction
        while pred[t] != sPos:
            t = pred[t]

        desired = self.directionTo(sPos,t)
        self.next[t.x][t.y] = 1
        
        return desired

class Calculator:

    def __init__(self, board: Board):
        self.CFG = board.configuration
        self.me = board.current_player_id
        print(self.me)
        self.playerNum = len(board.players)

    def update(self, board: Board):
        # Updates
        self.board = board

        # Encoding
        self.encode()

        # Calculate
            #Mean Halite 
        self.haliteMean = np.mean(self.haliteMap, axis=None)
            #Friendly units
        self.ally = self.shipMap[self.me]
            #Friendly shipyards
        self.allyShipyard = self.shipyardMap[self.me]
            #Enemy units
        self.enemy = np.sum(self.shipMap, axis=0) - self.ally
            #Enemy shipyards
        self.enemyShipyard = np.sum(self.shipyardMap, axis=0) - self.allyShipyard
            #Enemy units, encoded by halite. Empty is infinity.
        self.enemyShipHaliteMap()
            #Map control map
        self.controlMap = Calculator.controlMap(self.ally-self.enemy,self.allyShipyard-self.enemyShipyard)
            #Halite vicinity map
        self.haliteSpreadMap()

        #Ship specific calculations
        ships = board.current_player.ships
        shipyards = board.current_player.shipyards

        self.closestShipyardMap(shipyards,ships)

        self.shipMaps = {}
        for ship in ships:
            self.shipMaps[ship] = {}

            #Obstacle
            self.shipMaps[ship]['blocked'] = self.getAvoidanceMap(ship)


    # Encodes halite and units to matrices
    def encode(self) -> dict:
        # Map
        self.haliteMap = np.zeros((self.CFG.size, self.CFG.size))
        self.shipMap = np.zeros((self.playerNum, self.CFG.size, self.CFG.size))
        self.shipyardMap = np.zeros((self.playerNum, self.CFG.size, self.CFG.size))
        for cell in self.board.cells.values():
            self.haliteMap[cell.position.x][cell.position.y] = cell.halite
        for ship in self.board.ships.values():
            self.shipMap[ship.player_id][ship.position.x][ship.position.y] = 1
        for shipyard in self.board.shipyards.values():
            self.shipyardMap[shipyard.player_id][shipyard.position.x][shipyard.position.y] = 1

        # TODO: Add encoding for individual ships and yards (not necessary now)
    
    # Calculations
    
    def enemyShipHaliteMap(self):
        self.enemyShipHalite = np.zeros((self.CFG.size, self.CFG.size))
        self.enemyShipHalite += np.Infinity
        for ship in self.board.ships.values():
            if ship.player_id != self.me:
                self.enemyShipHalite[ship.position.x][ship.position.y] = ship.halite
    
    def closestShipyardMap(self,shipyards,ships):
        self.closestShipyard = [[None for y in range(self.CFG.size)]for x in range(self.CFG.size)]
        if len(shipyards) == 0:
            shipyards = [max(ships,key=lambda ship : ship.halite)]
        for x in range(self.CFG.size):
            for y in range(self.CFG.size):
                minimum = math.inf
                for shipyard in shipyards:
                    if nav.dist(Point(x,y),shipyard.position) < minimum:
                        minimum = nav.dist(Point(x,y),shipyard.position)
                        self.closestShipyard[x][y] = shipyard
        
        

    #Generate a control map. ships : unit map of allies / opponents
    @staticmethod
    def controlMap(ships,shipyards):
        
        ITERATIONS = 4
        STRENGTH = 0.3
        
        res = ships

        #TODO: Use convolutions instead of this hacky method.
        # Convolutions will be more extensible down the line

        for i in range(ITERATIONS):
            temp = np.roll(res,1,axis=0)
            temp += np.roll(res,-1,axis=0)
            temp += np.roll(res,1,axis=1)
            temp += np.roll(res,-1,axis=1)
            temp = temp * STRENGTH

            res += temp
        
        return res + shipyards

    #Generate a map with halite in the vicinity
    def haliteSpreadMap(self):
        self.haliteSpread = np.copy(self.haliteMap)
        for i in range(3):
            self.haliteSpread += np.roll(self.haliteMap,i,axis=0) / (i+1)
            self.haliteSpread += np.roll(self.haliteMap,-i,axis=0) / (i+1)
        temp = self.haliteSpread.copy()
        for i in range(3):
            self.haliteSpread += np.roll(temp,i,axis=1) / (i+1)
            self.haliteSpread += np.roll(temp,-i,axis=1) / (i+1)

    def getAvoidanceMap(self,s : Ship): #Returns a boolean array of where a ship cannot go

        threshold = s.halite

        #Enemy units
        temp = np.where(self.enemyShipHalite < threshold, 1, 0)
        enemyBlock = np.copy(temp)
        enemyBlock = enemyBlock + np.roll(temp,1,axis=0)
        enemyBlock = enemyBlock + np.roll(temp,-1,axis=0)
        enemyBlock = enemyBlock + np.roll(temp,1,axis=1)
        enemyBlock = enemyBlock + np.roll(temp,-1,axis=1)

        enemyBlock = enemyBlock + self.enemyShipyard - self.allyShipyard*5

        blocked = enemyBlock
        blocked = np.where(blocked>0,1,0)
        return blocked

# %% [markdown]
# # Agent

# %%
#More helper functions

def score(ship, cell,board):
    # Setup
    res = 0
    cfg = nav.CFG

    #TUNE
    HASTE_MULT = 200
    ALLY_MULT = 4
    OPP_MULT = 50
    STILL_MULT = 100
    INVENTORY_MULT = 0.05

    # https://www.kaggle.com/solverworld/optimal-mining-with-carried-halite

        # Calculate total travel distance
    travelDistance = nav.dist(ship.position,cell.position) + nav.dist(cell.position,calc.closestShipyard[cell.position.x][cell.position.y].position)

        # Starting Halite
    C = ship.halite
    H = cell.halite

        # Finding the highest by using a loop and mine at maximum 10 steps
    for M in range(1,11):
        res = max(res,(C+(1-0.75**M)*H)/(travelDistance+M) * HASTE_MULT) 
    
    # Intuitively, penalize areas where crowding has occured
    control = calc.controlMap[cell.position.x,cell.position.y] 

    if control <= 0:
        res += control * OPP_MULT
    else:
        res -= control * ALLY_MULT * nav.dist(ship.position,cell.position)

    # Support finish mining a spot
    if travelDistance == 0:
        res += STILL_MULT

    # Penalize if a ship is on the target or opponent guarding
    if cell.ship != None and cell.ship != ship:
        res -= 10000
    for target in nav.getAdjacent(ship.position):
        if board.cells[target].ship != None:
            targetShip = board.cells[target].ship
            if targetShip.player.id != calc.me and targetShip.halite <= ship.halite:
                res -= 10000

    # Penalize for Halite on board to urge ships to go back to base
    res -= INVENTORY_MULT * ship.halite * ship.halite

    return res

def shipyardScore(cell):
    #TODO: Tune
    SHIPYARD_DIST_MULT = 2000
    POSITION_HALITE_MULT = 10
    ALLY_MULT = 100
    OPP_MULT = 700
    pos = cell.position
    #How much Halite nearby
    res = calc.haliteSpread[pos.x][pos.y]
    #Score penalty for being close to friendly shipyard
    if calc.closestShipyard[pos.x][pos.y] != None:
        if (nav.dist(calc.closestShipyard[pos.x][pos.y].position,pos) ** 2) != 0:
            res -= 1 / (nav.dist(calc.closestShipyard[pos.x][pos.y].position,pos) ** 2) * SHIPYARD_DIST_MULT
        else:
            res = -10000
    #Try not to build on halite itself
    res -= calc.haliteMap[pos.x][pos.y] * POSITION_HALITE_MULT
    
    #Ally bonus, opponent penalty
    control = calc.controlMap[pos.x][pos.y] 
    if control <= 0:
        res += control * OPP_MULT
    else:
        res += max(400,control * ALLY_MULT)

    return res
    
@board_agent
def agent(board):
    global nav, calc
    
    '''
    if board.step > 5:
        #Stupid way of stopping the game
        10/0
    '''

    if board.step == 0:
        nav = Navigation(board)
        calc = Calculator(board)

    # Process map
    calc.update(board)
    nav.update()
    ships = board.current_player.ships
    shipyards = board.current_player.shipyards
    currentHalite = board.current_player.halite

    # Tasks
    action = {}

    #Convertion (Much to be improved)

        # 1. Convert max-halite ship if no shipyards remaining
        # TODO: Make max-halite ship go to a better position
    if len(shipyards) == 0:
        maxShip = max(ships,key=lambda ship : ship.halite)
        action[maxShip] = (math.inf,maxShip,maxShip.position,"convert")
        # 2. At maximum have 3 shipyards
    elif len(shipyards) <= 2:
        maxBuild = max(board.cells.values(),key=lambda cell : shipyardScore(cell))
        scoreBuild = shipyardScore(maxBuild)

        if scoreBuild > 2500 and len(ships) > 10 and currentHalite > 500:
            #Save 500 halite
            currentHalite -= 500
            closest=min(ships,key=lambda ship : nav.dist(maxBuild.position,ship.position))
            action[closest] = (math.inf,closest,maxBuild.position,"convert")


    #Ship action (Much to be improved)

        # 0. Setup

    #Consider only cells with halite above a threshold to reduce computation time
    miningCells = np.where(calc.haliteMap > calc.haliteMean / 1.5, 1, 0)
    miningIndex = np.argwhere(miningCells)
    taken = np.zeros((calc.CFG.size,calc.CFG.size))
    
    #Calculate current efficiency of mining

    #Estimate priority

    ships.sort(reverse=True,key=lambda ship : calc.haliteSpread[ship.position.x][ship.position.y])
    meanScore = 0
    miningShips = 0

    for ship in ships:
        # Already has task
        if ship in action:
            continue

        # 1.Attack nearby enemy
        for target in nav.getAdjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != calc.me and targetShip.halite > ship.halite:
                    action[ship] = (100,ship,targetShip.position,"attack")
        
        # Already has task
        if ship in action:
            continue

        # 2. Return if under attack

        for target in nav.getAdjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != calc.me and targetShip.halite < ship.halite:
                    action[ship] = (100000,ship,calc.closestShipyard[ship.position.x][ship.position.y].position,"return")

        if ship in action:
            continue

        # 3.Greedy mine

            # Rational of using greedy over the bipartite assignment problem: 
                # Easy implementation
                # More efficient
                # Near optimal solution

                # Cost heuristic is a modified 
                # https://www.kaggle.com/solverworld/optimal-mining-with-carried-halite
                # Accounting for map control and nearby units

                # TODO: Requires tuning / changing cost heuristic

        miningShips += 1

        finalChoice = None          #The chosen greedy target
        maximumScore = -math.inf    #The score of the greedy target. 
        for x,y in miningIndex:
            if taken[x][y]:
                continue
            targetCell = board.cells[Point(x,y)]
            v = score(ship,targetCell,board)
            if v > maximumScore:
                maximumScore = v
                finalChoice = targetCell
        
        meanScore += maximumScore

        #Don't mine if score too low. Else, go back to base.
        MINE_CUTOFF_SCORE = -400
        if board.step > 390 and ship.halite > 50:
            action[ship] = (ship.halite,ship,calc.closestShipyard[ship.position.x][ship.position.y].position,'return')
        elif maximumScore > MINE_CUTOFF_SCORE:
            action[ship] = (maximumScore,ship,finalChoice.position,"mine")
            taken[finalChoice.position.x][finalChoice.position.y] = 1
        elif ship.halite > 0:
            action[ship] = (ship.halite,ship,calc.closestShipyard[ship.position.x][ship.position.y].position,'return')
        elif MINE_CUTOFF_SCORE > -5000:
            action[ship] = (maximumScore,ship,finalChoice.position,"mine")
            taken[finalChoice.position.x][finalChoice.position.y] = 1

        if ship in action:
            continue

        # 4. TODO: Add attack / patrol tasks if no target instead of nothing. For now, just try not to die.
        action[ship] = (-10000,ship,Point(random.randint(0,21),random.randint(0,21)),'move')
    
    if miningShips > 0:
        meanScore = meanScore / miningShips
         
    # Action processing
    
    action = list(action.values())
    action.sort(reverse=True,key=lambda x : x[0])
    for i in action:
        if i[-1] == "convert":
            if i[1].position == i[2]:
                i[1].next_action = ShipAction.CONVERT
            else:
                i[1].next_action = nav.safeMoveTo(i[1],i[2])
        elif i[-1] == "attack":
            i[1].next_action = nav.safeMoveTo(i[1],i[2])
        elif i[-1] == "mine":
            i[1].next_action = nav.safeMoveTo(i[1],i[2])
        elif i[-1] == "return":
            i[1].next_action = nav.safeMoveTo(i[1],i[2])
        else:
            i[1].next_action = nav.safeMoveTo(i[1],i[2])

    #Shipyard action (Much to be improved)

    createdNumber = 0
    #TODO: Tune
        # 1. Maintain 12 ships
    shipyards.sort(reverse=True,key=lambda shipyard : calc.haliteSpread[shipyard.position.x][shipyard.position.y])
    for shipyard in shipyards:
        if len(ships) + createdNumber < 12:
            if currentHalite > 500 and not nav.next[shipyard.cell.position.x][shipyard.cell.position.y]:
                shipyard.next_action = ShipyardAction.SPAWN   
                createdNumber += 1 
                currentHalite -= 500

        else:
        # 2. Maintain player Halite around equal to ship cost
            targetHaliteStore = (len(ships) + createdNumber - 12) * 500
            if currentHalite > max(500,targetHaliteStore) and not nav.next[shipyard.cell.position.x][shipyard.cell.position.y]:
                shipyard.next_action = ShipyardAction.SPAWN
                createdNumber += 1 
                currentHalite -= 500

# %% [markdown]
# # Run



# %%


