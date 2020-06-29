#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  14 09:45:29 2019

@author: bala
"""
import random
import numpy as np
import copy
from operator import itemgetter 
from sklearn.metrics import mean_squared_error as skMSE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class agentModelFC1(nn.Module):
    def __init__(self,env, device):
        super().__init__()
        self.numFeatures = env.getStateSpace()
        
        self.device = device
        
        self.l1 = nn.Linear(in_features = self.numFeatures, out_features = 1000)
        self.l2 = nn.Linear(in_features = 1000, out_features = 500)
        self.l3 = nn.Linear(in_features = 500, out_features = 25)
        self.l4 = nn.Linear(in_features = 25, out_features = len(env.getActionSpace()))

    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
        
class SimpleNNagent():
    def __init__(self,env):
        self.curState = []
        self.nxtState = []
        self.doneList = []
        self.rwdList = []
        self.actnList = []
        self.trainX = []
        self.trainY = []
        self.maxReplayMemory = 3000
        self.epsilon = 1.0
        self.minEpsilon = 0.1
        self.epsilonDecay = 0.9995
        self.discount = 0.95
        self.learningRate = 0.0000001
        self.batchSize = 32
        self.envActions = env.getActionSpace()
        self.nActions = len(self.envActions)
        self.buildModel(env)
        
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_CNN{random.randint(0, 1000)}")
        print(f"Log Dir: {self.sw.log_dir}")
        
    def buildModel(self,env):   
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device : {self.device}')
        self.model = agentModelFC1(env, self.device).to(self.device)
        self.loss_fn = nn.MSELoss()
#        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learningRate)
#        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.1)
        
    def trainModel(self):
        self.model.train()
        X = torch.from_numpy(self.trainX).to(self.device)
        Y = torch.from_numpy(self.trainY).to(self.device)
        for i in range(1): # number epoch
            self.optimizer.zero_grad()
            predY = self.model(X.float())
            loss = self.loss_fn(Y,predY)
            loss.backward()
            self.optimizer.step()
        
    def EpsilonGreedyPolicy(self,state):
        if random.random() <= self.epsilon:
            # choose random
            action = self.envActions[random.randint(0,self.nActions-1)]
        else:
            #ChooseMax
            #Handle multiple max
            self.model.eval()
            X = torch.from_numpy(np.reshape(state,(1,-1))).to(self.device)
            self.qValues = self.model(X.float()).cpu().detach().numpy()[0]
#            print(".............X..........", self.qValues)
            action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
        return action
    
    def getMaxAction(self, state):
        self.model.eval()
        X = torch.from_numpy(np.reshape(state,(1,-1))).to(self.device)
        self.qValues = self.model(X.float()).cpu().detach().numpy()[0]
#            print(".............X..........", self.qValues)
        action = np.random.choice(
                        np.where(self.qValues == np.max(self.qValues))[0]
                        )
        return action
    
    def newGame(self):
        self.trainX = []
        self.trainY = []
#        print("new game")
    
    def getTrainAction(self,state):
        action = self.EpsilonGreedyPolicy(state)
        return action    
    
    
    def buildReplayMemory(self, currentState, nextState, action, done, reward):
        if len(self.curState)> self.maxReplayMemory-1:
            # pop the first elemtnt of the list
            self.curState.pop(0)
            self.nxtState.pop(0)
            self.actnList.pop(0)
            self.doneList.pop(0)
            self.rwdList.pop(0)
        
        self.curState.append(currentState)
        self.nxtState.append(nextState)
        self.actnList.append(action)
        self.doneList.append(done)
        self.rwdList.append(reward)
    
    def buildMiniBatchTrainData(self):
        
        c = []
        n = []
        r = []
        d = []
        a = []
        
        if len(self.curState)>self.batchSize:
            ndxs = random.sample(range(len(self.curState)), self.batchSize)
        else:
            ndxs = range(len(self.curState))
       
        bSize = len(ndxs)   
        
        c = np.asanyarray(np.array(itemgetter(*ndxs)(self.curState)))
        n = np.asanyarray(np.array(itemgetter(*ndxs)(self.nxtState)))
        r = np.asanyarray(np.array(itemgetter(*ndxs)(self.rwdList)))
        d = np.asanyarray(np.array(itemgetter(*ndxs)(self.doneList)))
        a_ = np.array(itemgetter(*ndxs)(self.actnList))
        aTemp = np.vstack((np.array(range(len(a_))),a_))
        a = np.asanyarray(aTemp)
        

        self.model.eval()
        X = torch.from_numpy(np.reshape(n,(bSize,-1))).to(self.device)
        qVal_n = self.model(X.float()).cpu().detach().numpy()
        qMax_n = np.max(qVal_n, axis  = 1)
        X = torch.from_numpy(np.reshape(c,(bSize,-1))).to(self.device)
        qVal_c = self.model(X.float()).cpu().detach().numpy()
        Y = copy.deepcopy(qVal_c)
        y = np.zeros(r.shape)
        ndx = np.where(d == True)
        y[ndx] = r[ndx]
        ndx = np.where(d == False)
        y[ndx] = r[ndx] + self.discount * qMax_n[ndx]
        Y[a[0],a[1]] = y
        self.trainX = c
        self.trainY = Y

        return skMSE(Y,qVal_c)
        
    def saveModel(self, filePath):
        torch.save(self.model, f"{filePath}/{self.model.__class__.__name__}.pt")
    
    def loadModel(self, filePath):
        self.model = torch.load(filePath)
        self.model.eval()
    
    def formatInput(self, states):
        out = []
#        for state in states:
#            out.append(np.concatenate((state[0], state[1].flatten())))
        for state in states:
            out.append(state[1].flatten())
        return out
        
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, loss, reward, last100Rwd, lenEpisode):
        self.sw.add_scalar('5.Loss', loss, episode)
        self.sw.add_scalar('3.Reward', reward, episode)
        self.sw.add_scalar('4.Episode Length', lenEpisode, episode)
        self.sw.add_scalar('2.Epsilon', self.epsilon, episode)
        self.sw.add_scalar('1.Average of Last 100 episodes', last100Rwd, episode)
        
    def summaryWriter_close(self):
        self.sw.close()
    
    