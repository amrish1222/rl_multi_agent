# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import keyboard

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
from collections import defaultdict
from time import time as t

from ppo_agent import PPO
from ppo_agent import Memory
from constants import CONSTANTS
CONST = CONSTANTS()

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)

def getKeyPressOld(act):
    k = cv2.waitKeyEx(1) 
    #            print(k)
    if k == 2490368:
        act = 1
    elif k == 2424832:
        act = 2
    elif k == 2621440:
        act = 3
    elif k == 2555904:
        act = 4
    return act

def getKeyPress(act):
#    if keyboard.is_pressed('['):
#        act = 1
#    elif keyboard.is_pressed(']'):
#        act = 2
    return act

from obstacle import Obstacle
obsMaps = Obstacle(np.zeros((CONST.MAP_SIZE, CONST.MAP_SIZE)))

cur_num_agents = CONST.MAX_NUM_AGENTS

env = Env(obsMaps, cur_num_agents)

memory = Memory()
rlAgent = PPO(env, cur_num_agents)

#rlAgent.loadModel("checkpoints/ActorCritic_10000.pt", 1)

NUM_EPISODES = 30000
LEN_EPISODES = CONST.LEN_EPISODE
UPDATE_TIMESTEP = 1000
curState = []
newState= []
reward_history = []
agent_history_dict = defaultdict(list)
totalViewed = []
dispFlag = False

#curRawState = env.reset()
#curState = rlAgent.formatInput(curRawState)
#rlAgent.summaryWriter_showNetwork(curState[0])

keyPress = 0
timestep = 0
loss = None

for episode in tqdm(range(NUM_EPISODES)):
    cur_num_agents = np.random.randint(2, CONST.MAX_NUM_AGENTS + 1)
    rlAgent.change_num_agents(cur_num_agents)
    curRawState = env.reset(cur_num_agents)
    
    # generate state for each agent
    curState = rlAgent.formatInput(curRawState)
    
    episodeReward  = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    agent_episode_reward = [0]* cur_num_agents
    
    for step in range(LEN_EPISODES):
        timestep += 1
        
        # render environment after taking a step
        keyPress = getKeyPress(keyPress)
        
        if keyPress == 1:
            env.render()
        
        # TODO save video
        if episode%500 in range(1,6) and step%4 == 0:
            env.save2Vid(episode, step)

        # Get agent actions
        aActions = rlAgent.policy_old.act(curState, memory, cur_num_agents)

        
        # do actions
        newRawState  = env.step(aActions)
#        newRawState  = env.step([0])
        agent_pos_list, current_map_state, local_heatmap_list, minimap_list, local_reward_list, shared_reward, done = newRawState
        if step == LEN_EPISODES -1:
            done = True
        
        for agent_index in range(cur_num_agents):
            if CONST.isSharedReward:
                memory.rewards.append(shared_reward)
            else:
                memory.rewards.append(local_reward_list[agent_index])
            memory.is_terminals.append(done)
            
        
        # update nextState
        newState = rlAgent.formatInput(newRawState)
        
        if timestep % UPDATE_TIMESTEP == 0:
            loss = rlAgent.update(memory)
            memory.clear_memory()
            timestep = 0
        
        # record history
        
        for i in range(cur_num_agents):
            if CONST.isSharedReward:
                agent_episode_reward[i] += shared_reward
            else:
                agent_episode_reward[i] += local_reward_list[i]
        episodeReward += shared_reward
#        print(shared_reward, step)
        # set current state for next step
        curState = newState
        
        if done:
            break
        
    # post episode
    
    # Record history        
    reward_history.append(episodeReward)
    
    for i in range(cur_num_agents):
        agent_history_dict[i].append((agent_episode_reward[i]))
    
    
    # You may want to plot periodically instead of after every episode
    # Otherwise, things will slow
    rlAgent.summaryWriter_addMetrics(episode, loss, reward_history, agent_history_dict, LEN_EPISODES)
    if episode % 50 == 0:
        rlAgent.saveModel("checkpoints")

    if episode % 1000 == 0:
        rlAgent.saveModel("checkpoints", True, episode)
            
    
rlAgent.saveModel("checkpoints")
env.out.release()
        
            