# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import keyboard

import time

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


env = Env()

memory = Memory(CONST.NUM_AGENTS, CONST.LEN_EPISODE)
rlAgent = PPO(env)


NUM_EPISODES = CONST.NUM_EPISODES
LEN_EPISODES = CONST.LEN_EPISODE
UPDATE_TIMESTEP = CONST.UPDATE_STEP
curState = []
newState= []
reward_history = []
mapNewVisPenalty_history = defaultdict(list)
totalViewed = []
dispFlag = False

#curRawState = env.reset()
#curState = rlAgent.formatInput(curRawState)
#rlAgent.summaryWriter_showNetwork(curState[0])

keyPress = 0
timestep = 0
loss = None

for episode in tqdm(range(NUM_EPISODES)):
    curRawState = env.reset()
    
    # generate state for each agent
    curState = rlAgent.formatInput(curRawState)
    
    episodeReward  = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    
    for step in range(LEN_EPISODES):
        timestep += 1
        
        # render environment after taking a step
        keyPress = getKeyPress(keyPress)
        
        if keyPress == 1:
            env.render()
        
        # TODO save video
        if episode%500 in range(10,15) and step%4 == 0:
            env.save2Vid(episode, step)
#        a = t()
        # Get agent actions
# =============================================================================
#         for i in range(CONST.NUM_AGENTS):
#             action = rlAgent.policy.act(curState[i], memory,i)
#             aActions.append(action)
# =============================================================================
        aActions = rlAgent.policy.act(curState, memory, CONST.NUM_AGENTS)
#        b = t()
#        print("step: ", round(b-a,2))
        
        # do actions
        
        newRawState  = env.step(aActions)
        agent_pos_list, current_map_state, local_heatmap_list, reward, done = newRawState
        if step == LEN_EPISODES -1:
            done = True
        
        # record only once for all agents and retrieve after calculating discounted reward
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
            
        
        # update nextState
        newState = rlAgent.formatInput(newRawState)
        
        if timestep % UPDATE_TIMESTEP == 0:
            #a = time.time()
            loss = rlAgent.update(memory)
            #print("update time: ", round(1000*(time.time() - a),2))
            memory.clear_memory()
            timestep = 0
        
        # record history
        episodeReward += reward
        # set current state for next step
        curState = newState
        
        if done:
            break
        
    # post episode
    
    # Record history        
    reward_history.append(episodeReward)
    
    # You may want to plot periodically instead of after every episode
    # Otherwise, things will slow
    rlAgent.summaryWriter_addMetrics(episode, loss, reward_history, LEN_EPISODES)
    if episode % 50 == 0:
        rlAgent.saveModel("checkpoints")
            
    
rlAgent.saveModel("checkpoints")
env.out.release()
        
            