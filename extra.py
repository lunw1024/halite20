# Tool #
def dryMove(pos, direction):
    if direction == "NORTH":
        return ((pos[0] - 1) % size, pos[1])
    elif direction == "SOUTH":
        return ((pos[0] + 1) % size, pos[1])
    elif direction == "EAST":
        return (pos[0], (pos[1] + 1) % size)
    elif direction == "WEST":
        return (pos[0], (pos[1] - 1) % size)
    else:  # STATIONARY
        return pos
    
def distance_to(self, target):
    if type(target) is Position:
        col = abs(self.x - target.x)
        col = min(col, observe.SIZE - col)
        row = abs(self.y - target.y)
        row = min(row_d, observe.SIZE - row)
        return row + col

def direction_to(self, target):
    res = Direction.STATIONARY
    if target.x != self.x:
        if target.x > self.x and target.x - self.x < observe.SIZE / 2:
            res = Direction.EAST
        elif self.x > target.x and self.x - target.x > observe.SIZE / 2:
            res = Direction.EAST
        else:
            res = Direction.WEST
    elif target.y != self.y:
        if target.y > self.y and target.y - self.y < observe.SIZE / 2:
            res = Direction.NORTH
        elif self.y > target.y and self.y - target.y > observe.SIZE / 2:
            res = Direction.NORTH
        else:
            res = Direction.SOUTH
    return res
