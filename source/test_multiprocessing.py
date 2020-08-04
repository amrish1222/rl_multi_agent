# -*- coding: utf-8 -*-

from multiprocessing.pool import ThreadPool

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
import random as rand
import time
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
CONC_EPISODES = 10

act = 0 

for episode in tqdm(range(0, NUM_EPISODES, CONC_EPISODES)):
    env.reset()
    env.render()
    for step in range(LEN_EPISODES):
        # step agent
        actionSets = []
        for _ in range(CONC_EPISODES):
            actions = []
            for i in range(CONST.NUM_AGENTS):
#                if i == 0:
#                    actions.append(getKeyPress())
#                else:
#                    actions.append(0)
                actions.append(rand.randint(0,4))
            actionSets.append(actions)
            actionSets2 = actionSets
        a = time.time()
        with ThreadPool(5) as p:
            out = p.map(env.step, actionSets2)
#        env.step(actions)
        b = time.time()
        print("step: ", round(1000*(b-a),2))
        env.render()
        