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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self, num_agents):
        self.actions = defaultdict(list)
        self.states = defaultdict(list)
        self.logprobs = defaultdict(list)
        self.rewards = defaultdict(list)
        self.is_terminals = defaultdict(list)
        self.num_agents = num_agents
    
    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()

        # actor
        self.feature1 = nn.Sequential(
                    nn.Conv2d(1,16,(3,3),1,1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16,32,(3,3),1,1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32,32,(3,3),1,1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten()
                    )
        self.reg1 = nn.Sequential(
                    nn.Linear(3*3*32, 500),
                    nn.ReLU(),
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(env.get_action_space())),
                    nn.Softmax(dim=-1)
                )
        
        # critic
        self.feature2 = nn.Sequential(
                    nn.Conv2d(1,16,(3,3),1,1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16,32,(3,3),1,1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32,32,(3,3),1,1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten()
                    )
        self.reg2 = nn.Sequential(
                    nn.Linear(3*3*32, 500),
                    nn.ReLU(),
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )

        self.train()
        
    def action_layer(self, x1):
        x = self.feature1(x1)
#        x = torch.cat((x,x2), dim = 1)
        x = self.reg1(x)
        return x
    
    def value_layer(self, x1):
        x = self.feature2(x1)
#        x = torch.cat((x,x2), dim = 1)
        x = self.reg2(x)
        return x
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory, agent_index):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            save_state = state.unsqueeze(0)
            action_probs = self.action_layer(save_state.unsqueeze(0))
            dist = Categorical(action_probs)
            action = dist.sample()
            
            memory.states[agent_index].append(save_state)
            memory.actions[agent_index].append(action)
            memory.logprobs[agent_index].append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = torch.diag(dist.log_prob(action))
        action_logprobs = action_logprobs.view(-1,1)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
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
        for i in reversed(range(memory.num_agents)):
            for reward, is_terminal in zip(reversed(memory.rewards[i]), reversed(memory.is_terminals[i])):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                all_rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
#        all_rewards = torch.tensor(all_rewards).to(device)
#        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        all_rewards =np.array(all_rewards)
        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        
        
        minibatch_sz = 1000
        
        # concatenate all the states, actions, logprobs
        temp_states = []
        temp_actions = []
        temp_logprobs = []
        for i in range(memory.num_agents):
            temp_states += memory.states[i]
            temp_actions += memory.actions[i]
            temp_logprobs += memory.logprobs[i]
            
        mem_sz = len(temp_states)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            prev = 0
            for i in range(minibatch_sz, mem_sz+1, minibatch_sz):
                
#                print(prev,i, minibatch_sz, mem_sz)
                mini_old_states = temp_states[prev:i]
                mini_old_actions = temp_actions[prev:i]
                mini_old_logprobs = temp_logprobs[prev:i]
                mini_rewards = all_rewards[prev:i]
                
                # convert list to tensor
                old_states = torch.stack(mini_old_states).to(device).detach()
                old_actions = torch.stack(mini_old_actions).to(device).detach()
                old_logprobs = torch.stack(mini_old_logprobs).to(device).detach()
                rewards = torch.from_numpy(mini_rewards).float().to(device)
                
                prev = i
                
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
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

        return states[2]
    
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
