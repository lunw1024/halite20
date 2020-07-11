from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import random

environment = make("halite", configuration={"size": 21, "startingHalite": 24000}, debug=False)

def run(agents):
    if len(agents) in [1,2,4]:
        environment.reset(len(agents))
        environment.run(agents)
        a = environment.toJSON()
        f = open(str(random.randint(0,10000))+".json","a")
        f.write(a)
        f.close()
        return environment.toJSON()
    else:
        print("error")

