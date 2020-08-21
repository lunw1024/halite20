from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np
import math
from random import seed
import time
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
torch.set_printoptions(profile="short")

NUM_AGENTS = 4
BOARD_SIZE = 21
TURNS = 400
EPSILON = 1.0 
EPSILON_DECAY = 0.998
TRAINING_ITERATIONS = 2300
EPOCHS = 10
REPLACE_TARGET_INTERVAL = 10
LEARNING_RATE = 0.1
REPLAY_CAPACITY = 100000
WARM_START_SAMPLES = 32*20
BATCH_SIZE = 32
GAMMA = 0.99
PRINT_INTERVAL = 100
running_avg_reward = []
episode_rewards = []

#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = args[0]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


env = make('halite', configuration={"randomSeed": 5, "episodeSteps": TURNS, "size": BOARD_SIZE}, debug=True)
_ = env.reset(num_agents=NUM_AGENTS)

ACTIONS = [
    ShipAction.NORTH,
    ShipAction.EAST,
    ShipAction.SOUTH,
    ShipAction.WEST,
    ShipAction.CONVERT,
    #None #Disable None for training attack
]

def world_feature(board):
    size = board.configuration.size
    me = board.current_player
    
    ships = np.zeros((1, size, size))
    ship_cargo = np.zeros((1, size, size))
    shipyard = np.zeros((1, size, size))

    map_halite = np.array(board.observation['halite']).reshape(1, size, size)/1000

    for iid, ship in board.ships.items():
        ships[0, ship.position[1], ship.position[0]] = 1 if ship.player_id == me.id else -1
        ship_cargo[0, ship.position[1], ship.position[0]] = ship.halite/1000

    for iid, yard in board.shipyards.items():
        shipyard[0, yard.position[1], yard.position[0]] = 1 if yard.player_id == me.id else -1
    #Halite Spread
    halite_spread = np.copy(map_halite)
    for i in range(1,5):
        halite_spread += np.roll(map_halite,i,axis=0) * 0.5**i
        halite_spread += np.roll(map_halite,-i,axis=0) * 0.5**i
    temp = halite_spread.copy()
    for i in range(1,5):
        halite_spread += np.roll(temp,i,axis=1) * 0.5**i
        halite_spread += np.roll(temp,-i,axis=1) *  0.5**i
    #Total Halite
    halite_total = np.ones((1,size,size))*np.sum(ship_cargo)
    #Mean Halite
    halite_mean  = np.ones((1,size,size))*np.sum(ship_cargo)/(size**2)
#     global state
#     np.set_printoptions(precision=3)
#     state['configuration'] = board.configuration
#     state['me'] = board.current_player_id
#     state['playerNum'] = len(board.players)
#     state['memory'] = {}
#     state['currentHalite'] = board.current_player.halite
#     state['next'] = np.zeros((board.configuration.size,board.configuration.size))
#     state['board'] = board
#     state['memory'][board.step] = {}
#     state['memory'][board.step]['board'] = board
#     state['cells'] = board.cells.values()
#     state['ships'] = board.ships.values()
#     state['myShips'] = board.current_player.ships
#     state['shipyards'] = board.shipyards.values()
#     state['myShipyards'] = board.current_player.shipyards
#     N = state['configuration'].size

#     # Halite 
#     state['haliteMap'] = np.zeros((N, N))
#     for cell in state['cells']:
#         state['haliteMap'][cell.position.x][cell.position.y] = cell.halite
#     # Halite Spread
#     state['haliteSpread'] = np.copy(state['haliteMap'])
#     for i in range(1,5):
#         state['haliteSpread'] += np.roll(state['haliteMap'],i,axis=0) * 0.5**i
#         state['haliteSpread'] += np.roll(state['haliteMap'],-i,axis=0) * 0.5**i
#     temp = state['haliteSpread'].copy()
#     for i in range(1,5):
#         state['haliteSpread'] += np.roll(temp,i,axis=1) * 0.5**i
#         state['haliteSpread'] += np.roll(temp,-i,axis=1) *  0.5**i
#     # Ships
#     state['shipMap'] = np.zeros((state['playerNum'], N, N))
#     state['enemyShips'] = []
#     for ship in state['ships']:
#         state['shipMap'][ship.player_id][ship.position.x][ship.position.y] = 1
#         if ship.player_id != state['me']:
#             state['enemyShips'].append(ship)
#     # Shipyards
#     state['shipyardMap'] = np.zeros((state['playerNum'], N, N))
#     state['enemyShipyards'] = []
#     for shipyard in state['shipyards']:
#         state['shipyardMap'][shipyard.player_id][shipyard.position.x][shipyard.position.y] = 1
#         if shipyard.player_id != state['me']:
#             state['enemyShipyards'].append(shipyard)
#     # Total Halite
#     state['haliteTotal'] = np.sum(state['haliteMap'])
#     # Mean Halite 
#     state['haliteMean'] = state['haliteTotal'] / (N**2)
#     # Estimated "value" of a ship
#     #totalShips = len(state['ships'])
#     #state['shipValue'] = state['haliteTotal'] / state
#     state['shipValue'] = ship_value()
#     # Friendly units
#     state['ally'] = state['shipMap'][state['me']]
#     # Friendly shipyards
#     state['allyShipyard'] = state['shipyardMap'][state['me']]
#     # Enemy units
#     state['enemy'] = np.sum(state['shipMap'], axis=0) - state['ally']
#     # Enemy shipyards
#     state['enemyShipyard'] = np.sum(state['shipyardMap'], axis=0) - state['allyShipyard']
#     # Closest shipyard
#     state['closestShipyard'] = closest_shipyard(state['myShipyards'])
#     # Control map
#     state['controlMap'] = control_map(state['ally']-state['enemy'],state['allyShipyard']-state['enemyShipyard'])
#     state['negativeControlMap'] = control_map(-state['enemy'],-state['enemyShipyard'])
#     state['positiveControlMap'] = control_map(state['ally'],state['allyShipyard'])
#     #Enemy ship labeled by halite. If none, infinity
#     state['enemyShipHalite'] = np.zeros((N, N))
#     state['enemyShipHalite'] += np.Infinity
#     for ship in state['ships']:
#         if ship.player.id != state['me']:
#             state['enemyShipHalite'][ship.position.x][ship.position.y] = ship.halite
#     # Avoidance map (Places not to go for each ship)
#     for ship in state['myShips']:
#         state[ship] = {}
#         state[ship]['blocked'] = get_avoidance(ship)
#         state[ship]['danger'] = get_danger(ship.halite)
#     # Who we should attack
#     if len(state['board'].opponents) > 0:
#         state['killTarget'] = get_target()
#     for i in state:
#         print(state[i])
#     return 1
    return np.concatenate([
        map_halite, 
        ships, 
        ship_cargo, 
        shipyard,
        halite_spread,
        halite_total,
        halite_mean
    ], axis=0)


#As example take the first frame of the game
sample_obs = env.state[0].observation
board = Board(sample_obs, env.configuration)

feature = world_feature(board)
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#q-network
class SmallModel(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(SmallModel, self).__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=3, #use this padding to give each pixel more vision
                padding_mode='circular'
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=0, #this will make the padded first layer smaller again
                padding_mode='circular'
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=0, #this will make the padded first layer smaller again
                padding_mode='circular'
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        
        self.output = nn.Linear(BOARD_SIZE*BOARD_SIZE*16, BOARD_SIZE*BOARD_SIZE*len(ACTIONS))
        
    def forward(self, features):
        x = self.network(features)
        x = x.view(features.shape[0], -1) #flatten
        x = self.output(x) #pass through linear layer
        
        return x.reshape(features.shape[0], self.num_actions, BOARD_SIZE, BOARD_SIZE)
                          

model = SmallModel(
    input_channels=7, #needs to be equal to the number of feature channels we have
    num_actions=len(ACTIONS)
)

target_model = SmallModel(
    input_channels=7, #needs to be equal to the number of feature channels we have
    num_actions=len(ACTIONS)
)

#predicting the feature from the cell above
feature_tensor = torch.from_numpy(feature).float().unsqueeze(0)
###
def make_move(model, obs, configuration, EPSILON):
    size = configuration.size
    board = Board(obs, configuration)
    me = board.current_player
    #if we do not have ships but a shipyard build 1 ship
    if len(me.ships)==0 and len(me.shipyards)>0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN
    #Random Spawn, needs improvement
    if len(me.ships)<=25:
        for shipyard in me.shipyards:
            i = np.random.randint(1,10)
            if i<=3:
                shipyard.next_action = ShipyardAction.SPAWN

    #if we have no shipyard build one
    state = world_feature(board).astype(np.float32)
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    
    action_indices = model(state_tensor).detach().numpy().argmax(1).squeeze()
    random_indices = np.random.choice(range(5), (size, size))
    actions = np.zeros((size, size))-1
    

    for ship in me.ships: 
        if len(me.shipyards)==0:
            action_index = -1
            ship.next_action = ShipAction.CONVERT #in our toy example we handle this manually
        else:
            if random.random() < EPSILON:
                action_index = random_indices[ship.position[1], ship.position[0]]
            else:
                action_index = action_indices[ship.position[1], ship.position[0]]
            
            ship.next_action = ACTIONS[action_index]
            
        actions[ship.position[1], ship.position[0]] = action_index
            
    return me.next_actions, state, actions

optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(model)
memory = ReplayMemory(REPLAY_CAPACITY)

for episode in range(TRAINING_ITERATIONS+1): #+1 so its inclusive and we print a statement at the end
    print(f'{episode} - {round(EPSILON,3)} - {len(memory)}', end='\r')
    _ = env.reset(num_agents=NUM_AGENTS)

    #When we call env.reset it will set the random seeds of both python and numpy to our fixed value
    #We want to do some real random exploration though, otherwise we will always end up with the same game
    seed_time = int(time.time()*1000)%1000000000
    np.random.seed(seed_time)
    seed(seed_time)
    size = env.configuration.size


    player2states = defaultdict(list)
    player2actions = defaultdict(list)

    player2halite = defaultdict(list)
    player2rewards = defaultdict(list)

    player2dones = defaultdict(list)

    #The gist of this loop is copied from Tom Van de Wiele's answer here: https://www.kaggle.com/c/halite/discussion/144844
    while not env.done:
        observation = env.state[0].observation
        player_mapped_actions = []
        for active_id in range(NUM_AGENTS):
            agent_status = env.state[active_id].status
            if agent_status == 'ACTIVE':
                player_obs = env.state[0].observation.players[active_id]
                observation['player'] = active_id
                engine_commands, state, actions = make_move(model, observation, env.configuration, EPSILON)

                player2states[active_id].append(state)
                player2actions[active_id].append(actions)

                #in the first round there was no previous reward, we will use 5000 so the diff is 0 and drop it later in post processing
                #one big thing is that we only generate one reward per frame, probably we should generate a (size, size) reward array to 
                #properly attribute the rewards to the ships if we have a multi ship scenario later
                prev_reward = 5000 if len(player2halite[active_id]) == 0 else player2halite[active_id][-1]
                reward = player_obs[0] - prev_reward
                reward = reward if reward > 0 else 0
                player2rewards[active_id].append(reward) 
                player2halite[active_id].append(player_obs[0]) 

                player2dones[active_id].append(env.done)

                player_mapped_actions.append(engine_commands)
            else:
                player_mapped_actions.append({})
        env.step(player_mapped_actions)


    #Postprocessing:
    #We need to build (state(t), actions(t), reward(t+1), state(t+1), dones(t+1)) tuples
    #After the env finished we want to set the last done to true
    #We want to add the last reward and remove the reward t=0 (since we always need reward(t+1))
    for active_id in range(NUM_AGENTS):
        player_obs = env.state[0].observation.players[active_id]
        player2dones[active_id][-1] = True #the main loop does not get called again when the env is done so we set it manually

        prev_reward = player2halite[active_id][-1]
        reward = player_obs[0] - prev_reward
        reward = reward if reward > 0 else 0
        player2rewards[active_id].append(reward) #append reward t+1
        player2rewards[active_id].pop(0) #remove reward t=0

        #For debugging: Make sure we have the same number of samples everywhere
        #print(len(player2states[active_id]), len(player2actions[active_id]),len(player2rewards[active_id]),len(player2dones[active_id]),)
        #Look at your rewards and compare with the replay below whether the reward matches the games that you see
        #print(player2rewards[active_id])


        states = player2states[active_id]
        next_states = [x for x in states]
        next_states = next_states[1:] + next_states[-1:]

        for state, action, reward, next_state, done in zip(states, player2actions[active_id], player2rewards[active_id], next_states, player2dones[active_id]):
            memory.push((state, action, reward, next_state, done))


    episode_rewards.append(np.array([x for y in player2rewards.values() for x in y]))

        
        
    running_avg_reward.append(episode_rewards[-1].sum()/episode_rewards[-1].shape)
    if episode % PRINT_INTERVAL == 0:
        episode_rewards = np.concatenate(episode_rewards)
        print(f'ep:{episode}, '
               f'mem_size:{len(memory)}, '
               f'rew:{episode_rewards.sum()}, '
               f'avg:{round(episode_rewards.sum()/episode_rewards.shape[0], 3)}, '
               f'eps:{round(EPSILON, 2)}, '
               f'running_avg_rew:{round(np.mean(running_avg_reward), 3)}'
              )
        episode_rewards = []
    
            
            
    if not len(memory)>WARM_START_SAMPLES:
        continue
        
    for epoch in range(EPOCHS):
        sample = memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). 
        # This converts [(state1, action1, reward1, next_state1, done1), (state2, action2, reward2, next_state2, done2)]
        # To:[(state1, state2), (action1, action2), (reward1, reward2), (next_state1, next_state2), (done1, done2)]
        states, actions, rewards, next_states, dones = list(zip(*sample))

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        #Its not a bad idea to check shapes again: states.shape, next_states.shape, actions.shape, rewards.shape, dones.shape
        states = torch.from_numpy(states)
        next_states = torch.from_numpy(next_states)
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones)

        #our actions are a (size, size) array, we want to select all fields that are not -1 since this is where a ship was that took an action
        #if we had mutliple ships then batch would contain the same frame multiple times with different x and y coordinates
        batch, xs, ys = np.where(actions>-1)

        taken_actions = actions[batch, xs, ys].unsqueeze(-1)

        #We will train multiple epochs, here you would ideally want to sample from a replay buffer

        current_qs = model(states)[batch, :, xs, ys].gather(1, taken_actions)
        next_qs = target_model(next_states).detach().max(1)[0][batch, xs, ys]

        # target_q = reward + 0.99 * next_state_max_q * (1 - done)
        target_qs = rewards[batch] + GAMMA * next_qs * ~dones[batch]
        loss = F.smooth_l1_loss(current_qs.squeeze(), target_qs.detach())
        #if we turn this on we will see that the loss is actually not decreasing very much from epoch to epoch
        #print(target_qs.shape, current_qs.shape, loss.mean(), end='\n')
        optimizer.zero_grad()
        loss.mean().backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    if episode and episode % REPLACE_TARGET_INTERVAL:
        target_model = copy.deepcopy(model)
        
    EPSILON *= EPSILON_DECAY 

torch.save(model, './model.h5')