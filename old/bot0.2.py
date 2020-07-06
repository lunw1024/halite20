# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Execution environment

# %%
print("Import started")
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import random
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

#TODO: Move CFG to static?


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

    def unpack(self, n):
        return Point(n // self.CFG.size, n % self.CFG.size)

    # Navigation
    def update(self):
        self.next = np.zeros((self.CFG.size,self.CFG.size))
    

    def safeMoveTo(self, s : Ship, t : Point): #A* Movement. Suggested move by priority.

        sPos = s.position

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


        threshold = s.halite
        enemyBlock = np.where(calc.enemyShipHalite <= threshold, 1, 0)
        enemyBlock = enemyBlock + calc.enemyShipyard
        blocked = self.next + enemyBlock
        blocked = np.where(blocked>0,1,0)
        #TODO: Improve obstacle calculation

        #Stay still
        if sPos == t:
            #Someone with higher priority needs position, must move
            if self.next[t.x][t.y]:
                for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
                    processPoint = sPos.translate(Point(offX,offY),self.CFG.size)
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
            for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
                processPoint = currentPoint.translate(Point(offX,offY),self.CFG.size)
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
            block = 0
            for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
                processPoint = sPos.translate(Point(offX,offY),self.CFG.size)
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
        # Swapping
        if calc.ally[t.x][t.y]:
            self.next[t.x][t.y] = 1
            pass
        
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
        self.haliteMean = np.mean(self.haliteMap, axis=None)
        self.ally = self.shipMap[self.me]
        self.allyShipyard = self.shipyardMap[self.me]
        self.enemy = np.sum(self.shipMap, axis=0) - self.ally
        self.enemyShipyard = np.sum(self.shipyardMap, axis=0) - self.allyShipyard
        self.enemyShipHaliteMap()

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

    def controlMap(self): # TODO: rename or refactor
        # TODO: Consider enemyShipHalite and shipyards
        self.controlMap = self.ally - self.enemy
        # TODO: avg pooling
    
    

# %% [markdown]
# # Agent

# %%
def cost(ship, cell):
    # TODO: much to improve
    # We can probably RL this
    cfg = environment.configuration
    haliteCoef = cfg.size / cfg.maxCellHalite
    return nav.dist(ship.position, cell.position) - haliteCoef * cell.halite

@board_agent
def agent(board):
    global nav, calc

    if board.step == 0:
        init = True
        nav = Navigation(board)
        calc = Calculator(board)

    # Process map
    calc.update(board)
    nav.update()
    ships = board.current_player.ships
    shipyards = board.current_player.shipyards

    # Decide tasks 
    # (priority,ship,targetLocation, type)
    action = {}
    miningCells = calc.haliteMap

        # Terrible mining algorithm, should probably come up with something entirely new
    assign = []
    for i, ship in enumerate(ships):
        if ship.cell.halite >= calc.haliteMean:
            action[ship] = (900,ship,ship.cell.position,"mining")
        else:
            if ship.halite > 500 and len(shipyards) > 0:
                action[ship] = (1000,ship,shipyards[0].position,"return")
            else:
                assign.append(ship)

    miningCells = np.argpartition(miningCells, -len(assign),axis=None)[-len(assign):]
    miningCells = miningCells.tolist()
    miningCells = [board.cells[nav.unpack(i)] for i in miningCells]

    costMatrix = np.array([[cost(ship, cell) for ship in assign] for cell in miningCells])
    tasks, _ = linear_sum_assignment(costMatrix)
    for i, ship in enumerate(assign):
        action[ship] = (500-cost(ship,miningCells[tasks[i]]),ship,miningCells[tasks[i]].position,"mining")


    # Action process
    action = list(action.values())
    action.sort(reverse=True,key=lambda x : x[0])
    for i in action:
        i[1].next_action = nav.safeMoveTo(i[1],i[2])

    if len(shipyards) == 0:
        ships[0].next_action = ShipAction.CONVERT
    for shipyard in shipyards:
        if shipyard.cell.ship is None and not nav.next[shipyard.cell.position.x][shipyard.cell.position.y]:
            shipyard.next_action = ShipyardAction.SPAWN

# %% [markdown]
# # Run
