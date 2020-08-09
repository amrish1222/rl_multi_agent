# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.distributions import Categorical
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states1 = []
        self.states2 = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states1[:]
        del self.states2[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()

        # actor
        self.feature1 = nn.Sequential(
                    nn.Conv2d(1,16,(8,8),4,1),
                    nn.ReLU(),
                    nn.Conv2d(16,32,(4,4),2,1),
                    nn.ReLU(),
                    nn.Conv2d(32,32,(3,3),1,1),
                    nn.ReLU(),
                    nn.Flatten()
                    )
        self.reg1 = nn.Sequential(
                    nn.Linear(6*6*32 + 2, 500),
                    nn.ReLU(),
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(env.get_action_space())),
                    nn.Softmax(dim=-1)
                )
        
#        self.action_layer = nn.Sequential(
#                    nn.Conv2d(1,16,(8,8),4,1),
#                    nn.ReLU(),
#                    nn.Conv2d(16,32,(4,4),2,1),
#                    nn.ReLU(),
#                    nn.Conv2d(32,32,(3,3),1,1),
#                    nn.ReLU(),
#                    nn.Flatten(),
#                    nn.Linear(6*6*32, 256),
#                    nn.ReLU(),
#                    nn.Linear(256, len(env.getActionSpace())),
#                    nn.Softmax(dim=-1)
#                )
        
        # critic
        self.feature2 = nn.Sequential(
                    nn.Conv2d(1,16,(8,8),4,1),
                    nn.ReLU(),
                    nn.Conv2d(16,32,(4,4),2,1),
                    nn.ReLU(),
                    nn.Conv2d(32,32,(3,3),1,1),
                    nn.ReLU(),
                    nn.Flatten()
                    )
        self.reg2 = nn.Sequential(
                    nn.Linear(6*6*32 + 2, 500),
                    nn.ReLU(),
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )
        
#        self.value_layer = nn.Sequential(
#                    nn.Conv2d(1,16,8,4,1),
#                    nn.ReLU(),
#                    nn.Conv2d(16,32,4,2,1),
#                    nn.ReLU(),
#                    nn.Conv2d(32,32,3,1,1),
#                    nn.ReLU(),
#                    nn.Flatten(),
#                    nn.Linear(6*6*32, 256),
#                    nn.ReLU(),
#                    nn.Linear(256, 1)
#                )

        self.train()
        
    def action_layer(self, x1, x2):
        x = self.feature1(x1)
        x = torch.cat((x,x2), dim = 1)
        x = self.reg1(x)
        return x
    
    def value_layer(self, x1, x2):
        x = self.feature2(x1)
        x = torch.cat((x,x2), dim = 1)
        x = self.reg2(x)
        return x
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state1 = torch.from_numpy(state[0]).float().to(device)
        state2 = torch.from_numpy(state[1]).float().to(device) 
        action_probs = self.action_layer(state1.unsqueeze(0), state2.unsqueeze(0))
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states1.append(state1)
        memory.states2.append(state2)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()

    def act_test(self, state):
        state1 = torch.from_numpy(state[0]).float().to(device)
        state2 = torch.from_numpy(state[1]).float().to(device)
        action_probs = self.action_layer(state1.unsqueeze(0), state2.unsqueeze(0))
        action_probs= action_probs.detach().numpy()[0]
        action = np.random.choice(
            np.where(action_probs == np.max(action_probs))[0]
        )

        return action


    def evaluate(self, state, action):
        action_probs = self.action_layer(*state)
        dist = Categorical(action_probs)
        
        action_logprobs = torch.diag(dist.log_prob(action))
        action_logprobs = action_logprobs.view(-1,1)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(*state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, env):
        self.lr = 0.000002
        self.betas = (0.9, 0.999)
        self.gamma = 0.99 
        self.eps_clip = 0.2
        self.K_epochs = 4
        
        torch.manual_seed(2)
        
        self.policy = ActorCritic(env).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic(env).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.device = device
        
        self.MseLoss = nn.MSELoss()
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_CNN{random.randint(0, 1000)}")
        print(f"Log Dir: {self.sw.log_dir}")
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states1 = torch.stack(memory.states1).to(device).detach()
        old_states2 = torch.stack(memory.states2).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate((old_states1, old_states2), old_actions)
            
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
        self.policy_old.load_state_dict(self.policy.state_dict())
        return advantages.mean().item()
        
    def formatInput(self, states):
        out = []
#        for state in states:
#            out.append(np.concatenate((state[0], state[1].flatten())))
#        for state in states:
        display = states[1].reshape((1, states[1].shape[0], states[1].shape[1]))
        # single adversary position
        relativePos = states[0][0]
        out.append([display, relativePos])
        return out
    
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, loss, rewardHistory, agent_RwdDict, lenEpisode):
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
        
        for item in agent_RwdDict:
            title ='4. Agent ' + str(item+1)
            if len(agent_RwdDict[item]) >= 100:
                avg_agent_rwd=  agent_RwdDict[item][-100:]
            else:
                avg_agent_rwd =  agent_RwdDict[item]
            avg_agent_rwd = mean(avg_agent_rwd)

            self.sw.add_scalar(title,avg_agent_rwd, len(agent_RwdDict[item])-1)    
           
    def summaryWriter_close(self):
        self.sw.close()
        
    def saveModel(self, filePath):
        torch.save(self.policy.state_dict(), f"{filePath}/{self.policy.__class__.__name__}.pt")
    
    def loadModel(self, filePath, cpu = 0):

        if cpu == 1:
            self.policy.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
        else:
            self.policy.load_state_dict(torch.load(filePath))
        self.policy.eval()
    
    def plot_kernels(self,tensor, fig, num_cols=6):
        if not tensor.ndim==4:
            raise Exception("assumes a 4D tensor")
        num_kernels = tensor.shape[0]
        num_rows = 1+ num_kernels // num_cols
        fig = plt.figure(fig, figsize=(num_cols,num_rows))
        for i in range(tensor.shape[0]):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            ax1.imshow(tensor[i][0])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show(block = False)
        plt.pause(0.005)
        plt.show()
        
    def getKernels(self):
#        kernels = self.policy.feature1[0].weight.cpu().detach().numpy()
#        self.plot_kernels(kernels, 2, 16)
        
        count = 0
        num_kernels = 16+32+32
        
        fig = plt.figure(2, figsize=(16,10))
        for feat in [self.policy.feature1[0], self.policy.feature1[2], self.policy.feature1[4]]:
            wt = feat.weight.cpu().detach().numpy()
            for i in range(feat.weight.shape[0]):
                ax1 = fig.add_subplot(5,16,count+1)
                count += 1
                w = wt[i][0]
                low =w.min()
                high = w.max()
                w = (w-low)/high
                ax1.imshow(wt[i][0], cmap = 'gray')
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show(block = False)
        plt.show()
        plt.pause(0.005)