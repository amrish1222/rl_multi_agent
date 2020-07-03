# -*- coding: utf-8 -*-
# author Amrish Bakaran
# Copyright
# brief Environment class for the simulation

import numpy as np
import random
import cv2

from constants import CONSTANTS as K
CONST = K()
from agent import Agent
from obstacle import Obstacle
obsMap = Obstacle()

np.set_printoptions(precision=3, suppress=True)
class Env:
    def __init__(self):
        self.timeStep = CONST.TIME_STEP
        # getting obstacle maps and visibility maps
        self.obsMaps, self.vsbs, self.vsbPolys, self.numOpenCellsArr = self.initObsMaps_Vsbs()
        self.obstacle_map , self.vsb, self.vsbPoly, self.mapId, self.numOpenCells = self.setRandMap_vsb()

        # initialize environment with obstacles and agents
        self.obstacle_viewed, self.current_map_state, self.agents, self.agent_local_view_list = self.init_env(CONST.NUM_AGENTS, self.obstacle_map)
        # modified: decay rate:
        self.decay= 1
        # modified: cap the upperbound of penalty
        self.cap= 400


        #save video
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(f"checkpoints/cnn1.avi", self.fourcc, 50, (700, 700))





    
    def init_env(self, num_agents, obstacle_map):
        # unviewed = 0
        # viewed = 255
        # obstacle = 150
        # agent Pos = 100
        # adversary Pos = 200
        
        obstacle_viewed = np.copy(obstacle_map)
        
        #initialize agents at random location
        agents = []
        agents_local_view_list = []
        # get open locations
        x,y = np.nonzero(obstacle_map == 0)
        ndxs = random.sample(range(x.shape[0]), num_agents)

        for ndx in ndxs[:num_agents]:
            agents.append(Agent(x[ndx]+0.5, y[ndx]+0.5))
#            agents.append(Agent())

        agent_pos_list = [agent.getState()[0] for agent in agents]
        agent_g_pos_list = self.cartesian2grid(agent_pos_list)

        # update visibility for initial position
        for agent_pos, agent_g_pos in zip(agent_pos_list, agent_g_pos_list):
            obstacle_viewed = self.vsb.update_visibility_get_local(agent_pos, agent_g_pos,obstacle_viewed, self.vsbPoly)
#            agents_local_view_list.append(temp)

        current_map_state = self.update_map_at_pos(agent_g_pos_list, obstacle_viewed, 100)
        
        self.local_heatmap_list = self.get_local_heatmap_list(current_map_state, agent_g_pos_list)
        
        return obstacle_viewed, current_map_state, agents, agents_local_view_list

    def initObsMaps_Vsbs(self):
        return obsMap.getAllObs_vsbs(np.zeros((CONST.MAP_SIZE, CONST.MAP_SIZE)))

    def setRandMap_vsb(self):
        i = random.randint(0, len(self.obsMaps)-1)
        return self.obsMaps[i], self.vsbs[i], self.vsbPolys[i], self.numOpenCellsArr[i], i

    def cartesian2grid(self, pos):
        g_pos = np.floor(pos).astype(np.int)
        return g_pos

    def is_valid_location(self, pos, valid_map):
        g_pos = self.cartesian2grid(pos)
        # range check
        if 0 <= g_pos[0] < valid_map.shape[0] and 0<= g_pos[1] < valid_map.shape[1]:
            pass
        else:
            return False
        # availability check
        if valid_map[g_pos[0], g_pos[1]] == 0:
            return True
        else:
            return False

    
    def update_map_at_pos(self, g_pos, use_map, val):
        updated_map = np.copy(use_map)
        for pos in g_pos:
            updated_map[pos[0],pos[1]] = val
        return updated_map
    
    def step_agent(self, action_list):
        # have to decide on the action space
        # waypoints or velocity
        posOut = []
        velOut = []
        agent_pos_list = [agent.getState()[0] for agent in self.agents]
        agent_g_pos_list = self.cartesian2grid(agent_pos_list)

        for agent, action in zip(self.agents, action_list):
            vel = np.array([0,0])
            if action == 0:
                pass
            elif action == 1:
                vel[1] = 1
            elif action == 2:
                vel[0] = -1
            elif action == 3:
                vel[1] = -1
            elif action == 4:
                vel[0] = 1
            agent.setParams(vel)
            predNewState = agent.predNewState(self.timeStep)
            # check if agent in obstacle
            agent_pos_list = [agent.getState()[0] for agent in self.agents]
            agent_g_pos_list = self.cartesian2grid(agent_pos_list)
            obstacle_map_with_agents = self.update_map_at_pos(agent_g_pos_list, self.obstacle_map, 100)
            isValidPt = self.is_valid_location(predNewState, obstacle_map_with_agents)
            if isValidPt:
                agent.updateState(self.timeStep)
                curState = agent.getState()
                posOut.append(curState[0])
                velOut.append(curState[1])
            else:
                curState = agent.getState()
                posOut.append(curState[0])
                velOut.append(curState[1])
        return posOut, velOut
    
    def get_action_space(self):
        return [0,1,2,3,4]
    
    def step(self, action_list):
        agent_pos_list, agent_vel_list = self.step_agent(action_list)
        agent_g_pos_list = self.cartesian2grid(agent_pos_list)

        for indx, agent_pos in enumerate(agent_pos_list):
            self.current_map_state = self.vsb.update_visibility_get_local(agent_pos, agent_g_pos_list[indx],self.current_map_state, self.vsbPoly)
        
#        # TODO local view with only visibility
#        self.agent_local_view_list[indx]
        # Genearete currentStateMap with decay reward:

        # 150 is the wall, 255 (->0) (is newly viewed), initial unviewed is 0,  all pixels except wall is < 0, agent 100


        # If pixel <= 0 (not wall (150), agent (100), current viewed (255->0)) decrease by decay rate
        self.current_map_state = np.where((self.current_map_state <= 0), self.current_map_state - self.decay,
                                            self.current_map_state)

        # set current under viewed pixels(255) to 0, after decay complete
        self.current_map_state = np.where((self.current_map_state == 255), 0, self.current_map_state)

        # apply the lowerbound cap for the penalty
        self.current_map_state = np.where((self.current_map_state < (-1 * self.cap)), -1 * self.cap,
                                            self.current_map_state)


        # update position to get current full map
        self.current_map_state = self.update_map_at_pos(agent_g_pos_list, self.current_map_state, 100)
        
        self.local_heatmap_list = self.get_local_heatmap_list(self.current_map_state, agent_g_pos_list)
        
        reward = self.get_reward(self.current_map_state)
        
        done = False
        
        return agent_pos_list, self.current_map_state, self.local_heatmap_list, reward, done
    
    def reset(self):
        
        # need to update initial state for reset function
        self.obstacle_map , self.vsb, self.vsbPoly, self.mapId, self.numOpenCells = self.setRandMap_vsb()
        self.obstacle_viewed, self.current_map_state, self.agents, self.agent_local_view_list = self.init_env(CONST.NUM_AGENTS, self.obstacle_map)

        action_list = [0 for _ in range(CONST.NUM_AGENTS)]
        state = self.step(action_list)
        
        return state
    
    def heatmap_render_prep(self, heatmap):
        cap = self.cap
        heatmapshow = np.rot90(heatmap, 1)

        heatmapshow = np.where(heatmapshow == 150, 20, heatmapshow)
        heatmapshow = np.where(heatmapshow < 0, -1 * heatmapshow * 255 / cap, -1 * heatmapshow)
        heatmapshow = np.where(heatmapshow >= self.cap, 255, heatmapshow)

        heatmapshow = heatmapshow.astype(np.uint8)

        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        
        return heatmapshow
    
    def exploration_render_prep(self, explr_map):
        img = np.copy(explr_map)
        img = np.rot90(img,1)
        r = np.where(img==150, 255, 0)
        g = np.where(img==100, 255, 0)
        
        b = np.zeros_like(img)
        b_n = np.where(img==255, 100, 0)
        bgr = np.stack((b,g,r),axis = 2)
        bgr[:,:,0] = b_n
        return bgr
    
    def render(self):

        img = np.copy(self.current_map_state)

        reward_map = img

        """ initialize heatmap """
        
        full_heatmap = self.heatmap_render_prep(reward_map)
        full_heatmap = cv2.resize(full_heatmap,(700,700),interpolation = cv2.INTER_AREA)
        cv2.imshow("heatMap", full_heatmap)
        
        agent_views_list = []
# =============================================================================
#         for agent_indx, local_view in enumerate(self.agent_local_view_list):
#             
#             temp = self.exploration_render_prep(local_view)
#             
#             agent_views_list.append(temp)
# =============================================================================
        
        for agent_indx, local_view in enumerate(self.local_heatmap_list):
            
            temp = self.heatmap_render_prep(local_view)
            
            agent_views_list.append(temp)
        
        rows = []
        for j in range(CONST.RENDER_ROWS):
            rows.append(np.hstack((agent_views_list[j*CONST.RENDER_COLUMNS : (j+1) * CONST.RENDER_COLUMNS])))
        
        agent_views = np.vstack((rows))
        
        displayImg = cv2.resize(agent_views,(CONST.RENDER_COLUMNS* 200,CONST.RENDER_ROWS*200),interpolation = cv2.INTER_AREA)
        
#        displayImg = cv2.resize(agent_views_list[0],(200,200),interpolation = cv2.INTER_AREA)
        cv2.imshow("Agent Views", displayImg)
        
        cv2.waitKey(1)
    
    def get_reward(self, current_map):

        #sum up reward on all free pixels
        actualR = np.where((current_map<= 0), current_map, 0)
        curSumR = np.sum(actualR)


        return curSumR
    
    def get_local_heatmap_list(self, current_map, agent_g_pos_list):
        local_heatmap_list = []
        for g in agent_g_pos_list:
            r = int((CONST.LOCAL_SZ -1) /2)
            lx = int(max(0, g[1] - r))
            hx = int(min(CONST.MAP_SIZE, r + g[1] + 1))
            ly = int(max(0, g[0] - r))
            hy = int(min(CONST.MAP_SIZE, r + g[0] + 1))
            tempMask = np.zeros_like(current_map)
            tempMask[lx: hx , ly : hy] = 1
            
            local_view = np.ones((CONST.LOCAL_SZ,CONST.LOCAL_SZ)) * 150
            
            llx = int(lx - (g[1] - r))
            hhx = int(hx - g[1] + r)
            
            lly = int(ly - (g[0] - r))
            hhy = int(hy - g[0] + r)
            
            local_view[llx: hhx, lly: hhy] = current_map.T[lx: hx , ly : hy]
            local_heatmap_list.append(local_view.T)
        return local_heatmap_list


    def save2Vid(self, episode, step):

        img = np.copy(self.current_map_state)

        reward_map = img

        """ initialize heatmap """

        full_heatmap = self.heatmap_render_prep(reward_map)
        full_heatmap = cv2.resize(full_heatmap, (700, 700), interpolation=cv2.INTER_AREA)
        display_string = "Episode: " + str(episode) + " Step: " + str(step)
        full_heatmap = cv2.putText(full_heatmap, display_string, (20,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255) , 2, cv2.LINE_AA) 
        self.out.write(full_heatmap.astype('uint8'))

        