# -*- coding: utf-8 -*-

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
import random as rand

from constants import CONSTANTS as K
CONST = K()

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)

def getKeyPress():
    wait = True
    while(wait):
        k = cv2.waitKeyEx(1) 
        #            print(k)
        if k == 2490368:
            act = 1
            wait = False
        elif k == 2424832:
            act = 2
            wait = False
        elif k == 2621440:
            act = 3
            wait = False
        elif k == 2555904:
            act = 4
            wait = False
    return act


env = Env()

NUM_EPISODES = 100
LEN_EPISODES = 1000

act = 0 

for episode in tqdm(range(NUM_EPISODES)):
    env.reset()
    env.render()
    for step in range(LEN_EPISODES):
        # step agent
        actions = []
        for i in range(CONST.NUM_AGENTS):
#            if i == 0:
#                actions.append(getKeyPress())
#            else:
#                actions.append(0)
            actions.append(rand.randint(0,4))
        env.step(actions)
        env.render()
        