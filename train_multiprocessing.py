import multiprocessing
import time
import os
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import random
from functools import partial
import multiprocessing as mp
import math
import numpy as np
import os, glob
import sys, getopt
TRAIN_TARGET = 'geneticBot'
WEIGHT_SIZE = [4,2,2,1,2]
POP = 40  #POP SIZE SHOULD NOT EXCEED THE NUMBER OF CPU CORES
ITER = 1
STEP = 0.3  #TOO LARGE FOR TUNING
N_CPU = POP
def store_list_of_arr(arr,path):
    a = open(path,'w')
    for i in arr:
        a.write(" ".join(list(map(str,i.tolist()))))
        a.write("\n")
    a.close()
#TODO: ELO rating needed

def run(agents,size=21,steps=400,seed=1):
    env = make("halite", configuration={"size": size, "startingHalite": 24000,"episodeSteps": steps,'randomSeed':seed}, debug=False)
    env.reset(len(agents))
    env.run(agents)
    return env

def fitness_halite(env):
    state = env.state[0]
    reward = state.reward
    return reward

def double_fitness(agent, n, a):
    # Run 5 1v1 against comp1.py and take average fitness_halite
    res = 0
    #print(n,a)
    
    for randomSeed in range(a,n+a):
        print("Seed-",randomSeed)
        res += fitness_halite(run([agent,'old/py/badBotv1.0.py','old/py/geneticBotv1.3.py','old/py/geneticBotv1.2.py'],seed=randomSeed)) / n
    return res

def test_fitness(weights):
    return sum(weights)

init_weights = []

# Uniform crossover
def crossover(parent1,parent2):
    if parent1.shape != parent2.shape:
        print("Shapes must be the same!")
    result = parent1.copy()
    cross = np.random.choice([True,False],parent1.shape)
    result[cross] = parent2[cross]
    return result

# Uniform mutation by step
def mutation(target,step):
    target = target.astype('float64')
    res = target.copy()
    res += np.random.uniform(-step,step,res.shape)
    return res

def reset():
    files = glob.glob('trainweights/*')
    for f in files:
        os.remove(f)

def build(weights):
    store_list_of_arr(weights,TRAIN_TARGET+'/weights.txt')
    os.system("python3 build.py "+TRAIN_TARGET)

def convert(weights):
    # Converts a thing to a program readable list of arrays
    a = 0
    res = []
    for i in WEIGHT_SIZE:
        res.append(weights[a:a+i])
        a+=i
    return res

# Load all weights in trainweights
def load():
    res = []
    for filepath in glob.iglob('trainweights/*.txt'):
        file = open(filepath,'r')
        a = file.read()
        file.close()
        res.append(np.array(a.split()))
    return res

#build.py multiprocessors version
def multiprocess_build(agent,TRAIN_TARGET):
    '''
    Each cpu controls a chromosome
    '''
    id = os.getpid()
    #print("Start Building"+TRAIN_TARGET+". Processor: "+str(id)+"\n")
    path = "trainscripts/"
    if not os.path.exists(path):
        os.mkdir(path)    
    output = open(path+str(id)+"_"+TRAIN_TARGET+".py","w+")
    store_list_of_arr(convert(agent),path+str(id)+"_trainweights.txt")
    f = open(path+str(id)+"_trainweights.txt","r")
    a = f.read()
    f.close()
    a = a.rstrip()
    a = "weights='''"+a+"'''"
    a = a + '\n'
    output.write(a)
    f = open(TRAIN_TARGET+"/dependency.py","r")
    for line in f:
        output.write(line)
    output.write("\n")
    f.close()
    files = glob.glob(TRAIN_TARGET+"/*.py")
    for file in files:
        if file == TRAIN_TARGET + "/agent.py" or file == TRAIN_TARGET+"/dependency.py":
            continue
        f = open(file,"r")
        for line in f:
            if line.startswith('from') or line.startswith ('import'):
                continue
            output.write(line)
        output.write("\n")
        f.close()
    f = open(TRAIN_TARGET + "/agent.py","r")
    for line in f:
        if line.startswith('from') or line.startswith ('import'):
            continue
        output.write(line)
    f.close()
    output.close()

def job(workload):
    id = os.getpid()
    seed = workload[0]
    agent = workload[1]
    path = "trainscripts/"
    multiprocess_build(agent,TRAIN_TARGET)
    score = (double_fitness(path+str(id)+"_"+TRAIN_TARGET+".py",1,a))
    if score== 5000.0:
        score = 0    
    res = []
    res.append(score)
    res.append(agent)
    return res
if __name__ == '__main__':
    population = POP
    step = STEP
    iterations = ITER
    initial= load()
    N = sum(WEIGHT_SIZE)
    batch = None
    #TODO：optimize generation of initial chromosome (generate from best solutions)
    if initial != None:
        batch = initial
        a = 0
        if len(initial) != population:
            a = population - len(initial)
        for i in range(a):
            batch.append(np.random.uniform(-step*10,step*10,(N)))
    else:
        batch = np.array([np.random.uniform(-step*10,step*10,(N))for pop in range(population)])
    print("Start Training. Training population: %s Training iteration: %s" % (population,iterations))
    for i in range(iterations):
        print("========================")
        print("Iteration", i, "starting")
        if i % 1 == 0: #Tunable
            print("Saving all weights")
            reset()
            for j,agent in enumerate(batch):
                store_list_of_arr(convert(agent),'trainweights/'+str(j)+".txt")
        #Creating pool. Assigning work for cpus
        pool = mp.Pool(N_CPU)
        a = np.random.randint(1,100)
        workload = [[a] for _ in range(population)]
        for i in range(population):
            workload[i].append(batch[i])
        scores = []
        scores = pool.map(job,workload)
        pool.close()
        pool.join()
        scores.sort(reverse=True,key=lambda x:x[0])
        print("Maximum: ",max(scores,key=lambda x:x[0]))
        #Record the optimized value
        output = open("training_log.txt","w+")
        output.write(str(max(scores,key=lambda x:x[0])))
        output.close()
        # Take the top 25%
        top = population // 4
        stay = [x[1] for x in scores[0:top]]
        mutate = [mutation(x,step) for x in stay]
        cross = [crossover(random.choice(stay),random.choice(stay)) for x in stay]
        both = [mutation(crossover(random.choice(stay),random.choice(stay)),step) for x in stay]
        batch = stay + mutate + cross + both

