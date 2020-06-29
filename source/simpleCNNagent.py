#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  14 09:45:29 2019

@author: bala
"""
import random
import numpy as np
from statistics import mean 
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
        
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        
        self.fc1 = nn.Linear(in_features = (6*6*32), out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = len(env.getActionSpace()))

    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))
        
#        print(x.shape)
        
        x = x.flatten(start_dim = 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
class SimplecNNagent():
    def __init__(self,env):
        self.curState = []
        self.nxtState = []
        self.doneList = []
        self.rwdList = []
        self.actnList = []
        self.trainX = []
        self.trainY = []
        self.maxReplayMemory = 15000
        self.epsilon = 1
        self.minEpsilon = 0.1
        self.epsilonDecay = 0.9999769
        self.discount = 0.95
        self.learningRate = 0.0000001
        self.batchSize = 32
        self.envActions = env.getActionSpace()
        self.nActions = len(self.envActions)
        self.lr_decay = 0.9995
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
#        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma= self.lr_decay)
        
    def trainModel(self):
        self.model.train()
        X = self.trainX
        Y = self.trainY
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
            X = torch.from_numpy(np.reshape(state, (1,)+state.shape)).to(self.device)
            self.qValues = self.model(X.float()).cpu().detach().numpy()[0]
#            print(".............X..........", self.qValues)
            action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
        return action
    
    def getMaxAction(self, state):
        self.model.eval()
        X = torch.from_numpy(np.reshape(state, (1,)+state.shape)).to(self.device)
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
        
        self.curState.append(torch.from_numpy(currentState).to(self.device))
        self.nxtState.append(torch.from_numpy(nextState).to(self.device))
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
       
        
        c = itemgetter(*ndxs)(self.curState)
        n = itemgetter(*ndxs)(self.nxtState)
        r = np.asanyarray(np.array(itemgetter(*ndxs)(self.rwdList)))
        d = np.asanyarray(np.array(itemgetter(*ndxs)(self.doneList)))
        a_ = np.array(itemgetter(*ndxs)(self.actnList))
        aTemp = np.vstack((np.array(range(len(a_))),a_))
        a = np.asanyarray(aTemp)
        
        # sending current states and next states together for inference
        X = torch.stack(n+c)
        
        self.model.eval()
        
        qVal = self.model(X.float()).cpu().detach().numpy()
        
        # splitting them to get the current and next states
        hIndx = self.batchSize
        qVal_n = qVal[:hIndx]
        qMax_n = np.max(qVal_n, axis  = 1)
        qVal_c = qVal[hIndx:]

        Y = copy.deepcopy(qVal_c)
        y = np.zeros(r.shape)
        ndx = np.where(d == True)
        y[ndx] = r[ndx]
        ndx = np.where(d == False)
        y[ndx] = r[ndx] + self.discount * qMax_n[ndx]
        Y[a[0],a[1]] = y
        self.trainX = X[hIndx:]
        self.trainY = torch.from_numpy(Y).to(self.device)
        
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
            temp = state[1].reshape((1, state[1].shape[0], state[1].shape[1]))
            temp.shape
            out.append(temp)
        return out
        
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, loss, rewardHistory, mapRwdDict, lenEpisode):
        self.sw.add_scalar('6.Loss', loss, episode)
        self.sw.add_scalar('3.Reward', rewardHistory[-1], episode)
        self.sw.add_scalar('5.Episode Length', lenEpisode, episode)
        self.sw.add_scalar('2.Epsilon', self.epsilon, episode)
        
        if len(rewardHistory)>=100:
            avg_reward = rewardHistory[-100:]
            avg_reward = mean(avg_reward)
        else:    
            avg_reward = mean(rewardHistory) 
        self.sw.add_scalar('1.Average of Last 100 episodes', avg_reward, episode)
        
        for item in mapRwdDict:
            title ='4. Map ' + str(item + 1)
            if len(mapRwdDict[item]) >= 100:
                avg_mapReward,avg_newArea, avg_penalty =  zip(*mapRwdDict[item][-100:])
                avg_mapReward,avg_newArea, avg_penalty = mean(avg_mapReward), mean(avg_newArea), mean(avg_penalty)
            else:
                avg_mapReward,avg_newArea, avg_penalty =  zip(*mapRwdDict[item])
                avg_mapReward,avg_newArea, avg_penalty = mean(avg_mapReward), mean(avg_newArea), mean(avg_penalty)

            self.sw.add_scalars(title,{'Total Reward':avg_mapReward,'New Area':avg_newArea,'Penalty': avg_penalty}, len(mapRwdDict[item])-1)
            
    def summaryWriter_close(self):
        self.sw.close()
    
    