# References
Basics:  
https://www.kaggle.com/c/halite/overview/halite-rules  
https://www.kaggle.com/sam/halite-sdk-overview  

# Strategy

## Reward System: 
With each ship as an individual, a reward for each square is evaluated on each step. The total reward is then maximized through the Hungarian algorithm.
The reward for each square depends on the *tasks*, which can then be calcualed through the selection of features and running through a series of calculations with *weights* for each feature. The weights can be optimized in a training algorithm. 

For specific calculations, see reward.py.

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

## Spawning and convert: 

For both spawning and conversion, a "value" is calculated for the action. If the value is deemed higher than the halite cost of 500, conduct the action deemed higher. As with ships, values are calculated through features and weights with a series of calculations.

Spawning features
1. Board step
2. Opponent ships
3. My ships
4. Halite mean

Convertion features
1. Nearest shipyard (regardless of team) distance
2. Relative control
3. Halite spread
4. Ship shipyard ratio

## Endgame
In the last 30 or so steps, all ships with halite should return. Otherwise, suicide rush on a target.

## Navigation
Dijksta is used to move ships. 
Safety is ensured by giving a high *cost* to moving in dangerous areas. Each ship has *blocked* squares where it should never move to.
Collision prevention is done by moving higher priority ships first, and labelling a *next* matrix. The order of movement is adjusted if it would result in a collision.

Under immediate danger, predictions are made based on adjacent squares to determine the relative safest square. (To be improved)

## Danger Map / Control Map
Consider current ship cargo to filter the dangerous ship, then: ![image.png](https://i.postimg.cc/3x7qTCvM/image.png)  
Similar idea for control map
