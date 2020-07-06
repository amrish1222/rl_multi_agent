# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import random
from collections import defaultdict
import itertools
import embedding_graph as EMG


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_agents= 6

class Memory:
    def __init__(self, num_agents, steps):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.num_agents = num_agents
        self.length_episode = steps
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()


        # actor

        # added embedding layer
        self.embeding_layer1 = EMG.embedding_layer()
        # added fully connected graph generator
        self.graph1 = EMG.FC_graph(num_agents)
        # added GAT network: with #attention head = 3
        self.GAT1 = EMG.GAT(500, 3)




        self.reg1 = nn.Sequential(
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(env.get_action_space())),
                    nn.Softmax(dim=-1)
                )
        
        # critic
        # added embedding layer
        self.embeding_layer2 = EMG.embedding_layer()
        # added fully connected graph generator
        self.graph2 = EMG.FC_graph(num_agents)
        # added GAT network: with #attention head = 3
        self.GAT2 = EMG.GAT(500, 3)
        

        self.reg2 = nn.Sequential(
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )

        self.train()
        
    def action_layer(self, x1):
        # Generating embedding vectors: Convert input [6,1, 25,25] to embedding [6, 500] (N, dim) 1-D embedding vectors for each agents
        x= self.embeding_layer1(x1)
        # self.graph is fully connected graph
        self.graph1.ndata['x'] = x
        # run the graph convolution (attention) to get new feature x and graph
        x, self.graph1 = self.GAT1(self.graph1, self.graph1.ndata['x'])
        # get action distribution
        x = self.reg1(x)
        #print(EMG.get_att_matrix(self.graph1))
        #print(x)
        return x
    
    def value_layer(self, x1):
        # Generating embedding vectors: Convert input [6,1, 25,25] to embedding [6, 500] (N, dim) 1-D embedding vectors for each agents
        x = self.embeding_layer2(x1)
        # self.graph is fully connected graph
        self.graph2.ndata['x'] = x
        # run the graph convolution (attention) to get new feature x and graph
        x, self.graph2 = self.GAT2(self.graph2, self.graph2.ndata['x'])
        #print(EMG.get_att_matrix(self.graph1))
        #print(x)
        x = self.reg2(x)
        return x
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory, num_agents):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        
            action_list = []
            for agent_index in range(num_agents):
                memory.states.append(state[agent_index])
                memory.actions.append(action[agent_index])
                memory.logprobs.append(dist.log_prob(action[agent_index])[agent_index])
                action_list.append(action[agent_index].item())
        return action_list
    
    def evaluate(self, state, action):
        action_logprob_list = []
        state_value_list = []
        dist_entropy_list = []
        for i in range(0,len(state),6):
            #print(i)
            temp_state = state[i : i+6]
            temp_action = action[i :  i+6]
            action_probs = self.action_layer(temp_state)
            dist = Categorical(action_probs)

    #        action_logprobs = torch.diag(dist.log_prob(action))
            action_logprobs = dist.log_prob(temp_action)
            action_logprobs = action_logprobs.view(-1,1)
            dist_entropy = dist.entropy()

            state_value = self.value_layer(temp_state)

            action_logprob_list.append(action_logprobs)
            state_value_list.append(torch.squeeze(state_value))
            dist_entropy_list.append(dist_entropy)

        action_logprobs = torch.cat(action_logprob_list)
        state_value = torch.cat(state_value_list)
        dist_entropy = torch.cat(dist_entropy_list)

        return action_logprobs, state_value, dist_entropy
        
class PPO:
    def __init__(self, env):
        self.lr = 0.000002
        self.betas = (0.9, 0.999)
        self.gamma = 1.0
        self.eps_clip = 0.2
        self.K_epochs = 4
        
        torch.manual_seed(2)
        
        self.policy = ActorCritic(env).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
#        self.policy_old = ActorCritic(env).to(device)
#        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_CNN{random.randint(0, 1000)}")
        print(f"Log Dir: {self.sw.log_dir}")
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        all_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            all_rewards.insert(0, discounted_reward)
        
        # create rewards for all agents
        temp_rewards = np.array(all_rewards)
        all_rewards = np.repeat(temp_rewards, memory.num_agents)
        # Normalizing the rewards:
        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        

        minibatch_sz = memory.num_agents * memory.length_episode

        #print(len(temp_states))
        mem_sz = len(memory.states)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            prev = 0
            for i in range(minibatch_sz, mem_sz+1, minibatch_sz):
                
#                print(prev,i, minibatch_sz, mem_sz)
                mini_old_states = memory.states[prev:i]
                mini_old_actions = memory.actions[prev:i]
                mini_old_logprobs = memory.logprobs[prev:i]
                mini_rewards = all_rewards[prev:i]
                
                # convert list to tensor
                old_states = torch.stack(mini_old_states).to(device).detach()
                old_actions = torch.stack(mini_old_actions).to(device).detach()
                old_logprobs = torch.stack(mini_old_logprobs).to(device).detach()
                rewards = torch.from_numpy(mini_rewards).float().to(device)
                
                prev = i
                
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                #print(" evaluation complete")
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())
                    
                # Finding Surrogate Loss:
                advantages = rewards - state_values.detach()
                advantages = advantages.view(-1,1)
    #            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean() + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy.mean()
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
#        self.policy_old.load_state_dict(self.policy.state_dict())
        return advantages.mean().item()
        
    def formatInput(self, states):
        return np.array(states[2]).reshape((len(states[2]), 1, states[2][0].shape[0], states[2][0].shape[1]))
    
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, loss, rewardHistory, lenEpisode):
        if loss:
            self.sw.add_scalar('6.Loss', loss, episode)
        self.sw.add_scalar('3.Reward', rewardHistory[-1], episode)
        self.sw.add_scalar('5.Episode Length', lenEpisode, episode)
        
        if len(rewardHistory)>=100:
            avg_reward = rewardHistory[-100:]
            avg_reward = mean(avg_reward)
        else:    
            avg_reward = mean(rewardHistory) 
        self.sw.add_scalar('1.Average of Last 100 episodes', avg_reward, episode)
#        
#        for item in mapRwdDict:
#            title ='4. Map ' + str(item + 1)
#            if len(mapRwdDict[item]) >= 100:
#                avg_mapReward,avg_newArea, avg_penalty, avg_totalViewed=  zip(*mapRwdDict[item][-100:])
#            else:
#                avg_mapReward,avg_newArea, avg_penalty, avg_totalViewed =  zip(*mapRwdDict[item])
#            avg_mapReward,avg_newArea, avg_penalty, avg_totalViewed = mean(avg_mapReward), mean(avg_newArea), mean(avg_penalty), mean(avg_totalViewed)
#
#            self.sw.add_scalars(title,{'Total Reward':avg_mapReward,'New Area':avg_newArea,'Penalty': avg_penalty, 'Total Viewed': avg_totalViewed}, len(mapRwdDict[item])-1)
#            
    def summaryWriter_close(self):
        self.sw.close()
        
    def saveModel(self, filePath):
        torch.save(self.policy, f"{filePath}/{self.policy.__class__.__name__}.pt")
    
    def loadModel(self, filePath):
        self.model = torch.load(filePath)
        self.model.eval()
