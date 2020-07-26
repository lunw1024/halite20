# Not used yet

shipToPatrol = {}
patrols = []

def patrol_tasks():
    global shipToPatrol
    shipToPatrol = {}
    for patrol in patrols:
        patrol.try_update_ship
    for patrol in patrols:
        patrol.update()

class Patrol:

    def init(self,path,priority):
        self.path = path
        self.shipID = None
        self.priority = priority
        self.reverse = False

    # Assign nearest empty ship
    def assign(self):
        emptyShips = []
        for ship in state['myShips']:
            if ship.halite != 0 or ship in shipToPatrol.keys():
                continue
            emptyShips.append(ship)
        if emptyShips == None:
            return

        self.shipID = closest_thing(self.path[0],emptyShips)
        ship = state['board'].ships[self.shipID]
        shipToPatrol[ship] = self

    def try_update_ship(self):
        if self.shipID in state['board'].ships.keys():
            shipToPatrol[state['board'].ships[self.shipID]] = self  


    def update(self):
        if not state['board'].ships[self.shipID] in shipToPatrol.keys():
            self.assign()
        if not state['board'].ships[self.shipID] in shipToPatrol.keys():
            print("No ship found for patrol task")
            return

        ship = state['board'].ships[self.shipID]
        # Should ship be in patrol
        if ship.halite != 0:
            self.shipID = None
            shipToPatrol.pop(ship)
            self.assign()
            if not state['board'].ships[self.shipID] in shipToPatrol.keys():
                print("No ship found for patrol task")
                return
            ship = state['board'].ships[self.shipID]

        if not ship.position in self.path:
            action[ship] = (self.priority,ship,closest_thing_position(ship.position,self.path))
            return
        
        if ship.position == self.path[-1]:
            self.reverse = True
        elif ship.position == self.path[0]:
            self.reverse = False

        if not self.reverse:
            action[ship] = (self.priority,ship,self.path[self.path.index(ship.position)+1])
        else:
            action[ship] = (self.priority,ship,self.path[self.path.index(ship.position)-1])



        

        
