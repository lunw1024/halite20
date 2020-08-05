## Navigation (navigation.py)

**d_move(s: Ship, t: Point, blocked)** -> The ShipAction to move ship s to t.
Each *Point* is viewed as a node on the graph. The cost of moving to a node is defined by *move_cost()*. *blocked* are the nodes that cannot be accessed (immediate danger). Runs standard Dijkstra https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm from start to finish. 
- *state['next']* is used to keep track of the positions of ally ships next turn. Used to alter *Blocked*. Once implemented, swapping occurs naturally as current positions are always labeled safe unless already moved.
- Collision prevention is done through recursion (line 165) where if target is an ally who has not moved, and cannot move, *blocked* and *state['next']* are relabelled and d_move() is done again. If the ally can move, then mark the ally position as *blocked* for the ally and run d_move() on the ally - this will force the ally ship to move to another position. 
- If there is no path to target, try to move in general direction.

**micro_run(s: Ship)** -> ShipAction
Under immediate danger, predictions are made based on adjacent squares to determine the relative safest square. Move to a square that is empty and with higher friendly control (To be much improved)

**direction_to(s:Point, t:Point)** -> ShipAction from s to t.

**directions_to(s:Point, t:Point)** -> ShipAction s from s to t.

**dist(a: Point, b: Point)** -> Manhattan distance from a to b accounting for wrap.

**move_cost(s: Ship, t: Point)** -> Number
Returns the cost of all edges leading into node t. (See *d_move()*)
Penalizes danger. (See *state['danger']*). Penalizes adjacent ships with equal halite to reduce collisions.
A higher cost will discourage ships from transiting using node t.

**unpack(n)** -> Returns Point(n // N, n % N) where N is map size. Basically transforms integer to point.

**dry_move(s: Point, d: ShipAction)** -> Point that is s offset by d

**safe_naive(s: Ship, t: Point, blocked)** -> Basic movement from s to t that does not really account for obstacles. 
Run only through d_move() to ensure collision prevention and *state['next']* updates.

## core.py

**convert_tasks()**
Conversion is done by updating a list of both real and imaginary shipyards *state['closestShipyard']*. Each shipyard is viewed as real by ships. When the ship is at a shipyard but the shipyard does not actually exists (imaginary), it tries to build one.
Imaginary shipyards will be added if below
- 0th turn (immediately build)
- 0 shipyards (add imaginary shipyard and force closest ship to go to shipyard point)
- Estimated value of shipyard both above 500 and above the value of a new ship. 

**ship_tasks()**
Calculates the target position for each individual ship, and the priority for the ship to go to the target. Stored in *state['action']*

- Ship must try to return to closest shipyard under immediate danger.
- Ship should execute endgame strategy near end of game (return if has halite, else rush opponent)
- Else, execute reward system. 

A list of tasks to consider is generated, in the form of a list of (Cell, 'type'). Eg.(Cell, 'guard') means this is a special *guard* reward. (Cell, 'cell') signifies normal reward. 
With each ship as an individual, a reward for each square is evaluated on each step.
The reward of each square is calculated through *rewards.py*. (guard, attack, mine, return)
The total reward is then maximized through the Hungarian algorithm, and then each ship is matched with the cell that will generate the highest reward. 

- Finally, execute all target actions (move ship to target in *state['actions']* by sorting by priority and executing *process_action*

**process_action(act)**
Tries to move ship act[1] to point act[2], accounting for convertion specified in *convert_tasks*.

**spawn_tasks()** 
Loops each shipyard. Spawns if conditions below are met.
- There will not be a friendly ship at the shipyard's position next turn to avoid collision. As all ship actions have been processed, just check *state['next']*
- Enough halite in bank
AND ( 
- Estimated value of a ship is above 500
OR 
- Last shipyard remaining, no defender, enemy will attack
)

## dependency.py

*weights* -> list of np array of weights used in rewards.py calculations.

*state* -> Dictionary. Most important static variable, because it acts as container for every static variable.
Key -- value
'configuration' -> Configuration

'me' -> Current player ID

'playerNum' -> Number of players

'memory'[step] -> Past states at step step

'next' -> See *navigation.py*

'cells' -> Iterable of all the cells on board

'ships' -> Iterable of all the ships on board

'shipyards' -> Iterable of all the shipyards on board

'myShips' -> List of my ships

'enemyShips' -> List of enemy ships

'myShipyards' -> List of my shipyards

'enemyShipyards' -> List of enemy shipyards

'haliteMap' -> NP array. [x][y] represents halite at Point(x,y)

'haliteSpread' -> NP array. [x][y] represents nearby halite at Point(x,y). Through convolution (卷积）.

'shipMap' -> NP array. [player_id][x][y] true if player_id ship at Point(x,y)

'shipyardMap' -> NP array. [player_id][x][y] true if player_id shipyard at Point(x,y)

'haliteTotal' -> Number. Total halite.

'haliteMean' -> Number. Mean halite.

'controlMap' -> NP array. [x][y] represents control at Point(x,y). Through convolution. Positive means friendly has higher presence, negative means enemy.

'negativeControlMap' -> NP array. control map but ignores ally units.

ship['blocked'] -> NP array. ship[x][y] true if ship at immediate danger(can die next turn) at Point(x,y).

ship['danger'] -> NP array. [x][y] represents the danger for ship at Point(x,y). Through convolution. Basically control map specific to ship (halite threshold), and ignoring all allies.

'killTarget' -> Player. Who we want to kill. Decided based on asset value.


**init(board: Board)**
**update(board: Board)**

## process.py 
Updates *state*

## rewards.py

### Mining reward (ship -> cell)
Consider distance, halite spread, safety, and current bank.  
Try to mine in the rich and near area without being destroyed in the next step.

### Attack reward features
1. Area control
2. Distance
3. Cell halite
4. Whether ship/shipyard belongs to "target", a player selected.

### Return reward
1. Distance
2. Ship halite
3. Stored halite

### Guarding (Defending a shipyard)
1. Distance
2. Stored halite

### Spawn rewards
Spawning features
1. Board step
2. Opponent ships
3. My ships
4. Halite mean

### Convert reward
Convertion features
1. Nearest shipyard (regardless of team) distance
2. Relative control
3. Halite spread
4. Ship shipyard ratio

## Endgame
In the last 30 or so steps, all ships with halite should return. Otherwise, suicide rush on a target.

