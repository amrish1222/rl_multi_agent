# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:08:16 2020

@author: amris
"""

import numpy as np
from queue import PriorityQueue
import time
import sys
import skimage.measure
import copy
import random
from constants import CONSTANTS as K
CONST = K()

maxVal = sys.float_info.max
root2 = round(2**0.5,4)

from obstacle import Obstacle
obsMap = Obstacle()

class nodeInfo:
    def __init__(self,
                 _parentNodeIndex = -1,
                 _cost2Come = maxVal,
                 _currentPos = (-1.0,-1.0,-1.0)):
                     
        self.parentNodeIndex = _parentNodeIndex   # index in the list
        self.currentPos = _currentPos # tuple (x, y) 
        self.cost2Come = _cost2Come    
        self.visbility_sets = obsMap.getVisibilitySets(np.zeros((CONST.MAP_SIZE, CONST.MAP_SIZE)))

class Non_rl_agent:
    def __init__(self, num_agents):
        self.is_target_reached = [True]* num_agents
        self.targets = []
        
    def act(self, curState):
        agent_pos_list, current_map_state, local_heatmap_list, minimap_list, local_reward_list, shared_reward, done = curState
        
        if len(self.targets) == 0:
            for agent_pos in agent_pos_list:
                self.targets.append(tuple(agent_pos-np.array([0.5, 0.5])))
        
        hot_spots, all_hot_spots = self.get_hotspots(current_map_state, len(agent_pos_list))[:len(agent_pos_list)]
        target_chosen = set([])
        action_list = []
        
        all_hot_spots_set = set([tuple(l) for l in all_hot_spots])
        
        for agent_pos, agent_idx in zip(agent_pos_list, range(len(agent_pos_list))):
#            print(not self.targets[agent_idx] in all_hot_spots_set)
            if self.hasReachedGoal(tuple(agent_pos - np.array([0.5, 0.5])), self.targets[agent_idx]) or not self.targets[agent_idx] in all_hot_spots_set:
                dist_all = self.get_dist(hot_spots, agent_pos)
                dist_argsort = dist_all.argsort()
                agent_priority = hot_spots[dist_argsort]
                
                action  = 0
    #            print(agent_priority, agent_pos)
                for target_pos in agent_priority:
                    if not tuple(target_pos) in target_chosen:
                        target_chosen.add(tuple(target_pos))
                        aa = self.nearest_nonzero_idx_v2(current_map_state, int(target_pos[0]), int(target_pos[1]))
                        target_pos = np.array(aa)
                        path, success = self.get_path_next_pos(agent_pos, target_pos, current_map_state)
    #                    success = False
    
                        self.targets[agent_idx] = tuple(target_pos)
                        if success:
                            next_pos = path[-2]
                            dir_dist = np.array([next_pos[1] - int(agent_pos[1]),
                                                 int(agent_pos[0]) - next_pos[0],
                                                 int(agent_pos[1]) - next_pos[1],
                                                 next_pos[0] - int(agent_pos[0])])
                            
                            action = 1 + np.argmax(dir_dist)
                        else:
                            action = np.random.randint(0,6)
                        break
                action_list.append(action)
            else:
                path, success = self.get_path_next_pos(agent_pos, self.targets[agent_idx], current_map_state)
    #                    success = False
                target_pos = self.targets[agent_idx]
                action  = 0
                if success:
                    next_pos = path[-2]
                    dir_dist = np.array([next_pos[1] - int(agent_pos[1]),
                                         int(agent_pos[0]) - next_pos[0],
                                         int(agent_pos[1]) - next_pos[1],
                                         next_pos[0] - int(agent_pos[0])])
                    
                    action = 1 + np.argmax(dir_dist)
                else:
                    action = np.random.randint(0,6)
                action_list.append(action)
        return action_list
    
    def nearest_nonzero_idx_v2(self,a,x,y):
        tmp = a[x,y]
        if tmp == 150.0:
            pass
        r,c = np.nonzero(a!=150.0)
        a[x,y] = tmp
        min_idx = ((r - x)**2 + (c - y)**2).argmin()
        return r[min_idx], c[min_idx]
    
    def get_hotspots(self, current_map_state, num_agents):
        decay_map = np.where(current_map_state<=0, current_map_state, 0)
        
        decay_map *= -1
        
        window_sz = 1 #int(CONST.LOCAL_SZ/6)
        
        mini_decay = skimage.measure.block_reduce(decay_map, (window_sz,window_sz), np.max)
#        print(mini_decay)
        sorted_args = np.dstack(np.unravel_index(np.argsort(mini_decay.ravel()), (mini_decay.shape[0], mini_decay.shape[1])))
        
        sorted_args = np.flip(sorted_args[0], axis = 0)
        
        specific_args = []
        for arg in sorted_args:
            if mini_decay[arg[0], arg[1]] > 0:
                specific_args.append(arg)
        specific_args = np.array(specific_args).reshape(-1,2)
        all_pos_args = np.copy(specific_args)
#        specific_args = sorted_args#[:num_agents].astype('float64')
        
        main_args = []
        
        for arg in specific_args:
            if len(main_args)>0:
                away_from_all = True
                for m in main_args:
                    dist_ = self.cost2Come(m* window_sz,arg*window_sz)
                    if dist_ < CONST.LOCAL_SZ/2:
                        away_from_all = False
                        break
                if away_from_all:
                    main_args.append(arg)
                if len(main_args) == num_agents:
                        break
            else:
                main_args.append(arg)
        specific_args = np.array(main_args)
        pos_args = np.copy(specific_args)
        pos_args *= np.array([window_sz, window_sz])
        pos_args += np.array([int(window_sz/2), int(window_sz/2)])
        pos_args = np.where(pos_args > CONST.MAP_SIZE-1, CONST.MAP_SIZE-1, pos_args)
        return pos_args, all_pos_args
    
    def get_hotspot_visibility(self):
        pass

    def get_dist(self, hot_spots, agent_pos):
        out = []
        for hot_spot in hot_spots:
            out.append(np.linalg.norm(hot_spot - agent_pos))
            
        return np.array(out)
    
    def get_path_next_pos(self, start_pt, endPt, obs_map):
        startPt = np.copy(start_pt)
        startPt -= np.array([0.5, 0.5])
        
        startPt = copy.deepcopy(tuple(startPt))
        endPt = copy.deepcopy(tuple(endPt))
        
        # executes and returns path, time taken
        startTime = time.time()
        
        q = PriorityQueue()
        isGoalVisited = False
        visitedSet = set([])
        nodeInfoList = []
        pos2NodeIndex = dict()
        finalPath = []
        
        tempParentNode = -1
        tempCost2Come = self.cost2Come(startPt, endPt)
        tempCurPos = startPt
        
        tempNodeInfo = nodeInfo(tempParentNode, tempCost2Come, tempCurPos)
        nodeInfoList.append(tempNodeInfo)
        
        parentIndex = 0
        currentIndex = 0
        
        # add the first point to the queue
        q.put([nodeInfoList[parentIndex].cost2Come,parentIndex])
        visitedSet.add(tempCurPos)
        pos2NodeIndex[tempCurPos] = parentIndex
        
        while not q.empty():
            tempGetQ = q.get()
            
            parentIndex = tempGetQ[1]
            parentPos = nodeInfoList[parentIndex].currentPos
            parentCost2Come = nodeInfoList[parentIndex].cost2Come
            
            if self.hasReachedGoal(parentPos, endPt):
                isGoalVisited = True
                goalNode = parentIndex
                break
            
            nextNodes = self.getNextNodesAndCost(parentPos, (CONST.MAP_SIZE-1, CONST.MAP_SIZE-1))
            
            for node in nextNodes:
                if not node[0] in visitedSet:
                    if not self.isGridPtInObs(node[0], obs_map):
                        tempParentNode = parentIndex
                        tempCost2Come = parentCost2Come + node[1]
                        tempCurPos = node[0]
                        
                        tempNodeInfo = nodeInfo(tempParentNode, tempCost2Come, tempCurPos)
                        nodeInfoList.append(tempNodeInfo)
                        
                        currentIndex += 1
                        
                        q.put([round(nodeInfoList[currentIndex].cost2Come,4),currentIndex])
                        
                        visitedSet.add(node[0])
                        pos2NodeIndex[node[0]] = currentIndex
                        
                else:
                    if not self.isGridPtInObs(node[0], obs_map):
                        tempCost2Come =  parentCost2Come + node[1]
                        indexOfPos = pos2NodeIndex.get(node[0])
                        if tempCost2Come < nodeInfoList[indexOfPos].cost2Come:
                            tempParentNode = parentIndex
                            tempCurPos = node[0]
                            
                            tempNodeInfo = nodeInfo(tempParentNode, tempCost2Come, tempCurPos)
                            nodeInfoList[indexOfPos]=tempNodeInfo
        
        if isGoalVisited:
#            print("Path Found")
            nextParentNode = goalNode
            finalPath.append(endPt)
            while (nextParentNode != -1):
                nextParentNI = nodeInfoList[nextParentNode]
                nextParentNode = nextParentNI.parentNodeIndex
                finalPath.append(nextParentNI.currentPos)
        else:
            print("No Path available")
        
        endTime = time.time()
        totalTimeTaken = (endTime - startTime)
        
        if isGoalVisited:
            return finalPath, isGoalVisited
        else:
            return [], isGoalVisited
    def cost2Come(self, pt1, pt2):
        fromPtx, fromPty = pt1
        toPtx, toPty = pt2
        # calculate L2 distance
        cost = ((toPtx-fromPtx)**2 + (toPty-fromPty)**2)**0.5
        return cost
    
    def hasReachedGoal(self, gridPt, goalPt):        
        return gridPt == goalPt
#        d = ((gridPt[0] - goalPt[0])**2 + (gridPt[1] - goalPt[1])**2)**0.5
#        return d<=1
    
    def getNextNodesAndCost(self, pt, max_size):
        xc,yc = pt
        nearbyList = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if xc+i>=0 and yc+j >=0 and xc+i<=max_size[0] and yc+j<=max_size[1]:
                    if abs(i*j) == 1:
#                        temp = [(xc+i,yc+j),root2]
#                        nearbyList.append(temp)
                        pass
                    elif not(i == 0 and j == 0):
                        temp =[(xc+i,yc+j),1]
                        nearbyList.append(temp)
        return nearbyList
    
    def isGridPtInObs(self, node, obs_map):
        temp = None
        if obs_map[int(node[0]), int(node[1])] == 100:
            temp = random.choice([True, False])
        else:
            temp = False
        return obs_map[int(node[0]), int(node[1])] == 150 or temp 