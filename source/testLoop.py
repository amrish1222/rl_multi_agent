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

np.set_printoptions(threshold=np.inf, linewidth=1000, precision=3, suppress=True)


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
    if keyboard.is_pressed('['):
        act = 1
    elif keyboard.is_pressed(']'):
        act = 2
    return act


env = Env()

memory = Memory(CONST.NUM_AGENTS)
rlAgent = PPO(env)

rlAgent.loadModel("checkpoints/ActorCritic_10000.pt", 1)

NUM_EPISODES = 3
LEN_EPISODES = 2000
UPDATE_TIMESTEP = 6000
curState = []
newState = []
reward_history = []
agent_history_dict = defaultdict(list)
totalViewed = []
dispFlag = False

# curRawState = env.reset()
# curState = rlAgent.formatInput(curRawState)
# rlAgent.summaryWriter_showNetwork(curState[0])

keyPress = 1
timestep = 0
loss = None


#whether per episode:
per_episode= 1





for episode in tqdm(range(NUM_EPISODES)):
    curRawState = env.reset()

    # generate state for each agent
    curState = rlAgent.formatInput(curRawState)

    if per_episode:
        env.out = cv2.VideoWriter(f"test_V/testVideo" + str(episode + 1) + ".avi", env.fourcc, 50, (700, 700))



    for step in range(LEN_EPISODES):
        timestep += 1

        # render environment after taking a step
        keyPress = getKeyPress(keyPress)

        if keyPress == 1:
            env.render()


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

        newRawState = env.step(aActions)
        agent_pos_list, current_map_state, local_heatmap_list, minimap_list, local_reward_list, shared_reward, done = newRawState

        if step == LEN_EPISODES - 1:
            done = True


        # update nextState
        newState = rlAgent.formatInput(newRawState)

        # set current state for next step
        curState = newState



        if done:
            break



