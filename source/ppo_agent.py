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
from constants import CONSTANTS
import embedding_graph as EMG
import dgl


CONST = CONSTANTS()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self, num_agents):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.num_agents = num_agents
    
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
#        self.feature1 = nn.Sequential(
#                    nn.Conv2d(1,16,(3,3),1,1),
#                    nn.BatchNorm2d(16),
#                    nn.ReLU(),
#                    nn.MaxPool2d(2),
#                    nn.Conv2d(16,32,(3,3),1,1),
#                    nn.BatchNorm2d(32),
#                    nn.ReLU(),
#                    nn.MaxPool2d(2),
#                    nn.Conv2d(32,32,(3,3),1,1),
#                    nn.BatchNorm2d(32),
#                    nn.ReLU(),
#                    nn.MaxPool2d(2),
#                    nn.Flatten()
#                    )
#        self.reg1 = nn.Sequential(
#                    nn.Linear(3*3*32, 500),
#                    nn.ReLU(),
#                    nn.Linear(500, 256),
#                    nn.ReLU(),
#                    nn.Linear(256, len(env.get_action_space())),
#                    nn.Softmax(dim=-1)
#                )

        # added embedding layer (shared)
        self.embeding_layer = EMG.embedding_layer()
        # added fully connected graph generator
        self.graph = EMG.FC_graph(CONST.NUM_AGENTS)
        # added GAT network: with #attention head = 3
        self.GAT = EMG.GAT(2 * 2 * 32, 3)




        #storing attention matrix:
        self.attention_mat= None












        self.reg1 = nn.Sequential(
                    nn.Linear(2 * 2 * 64, 500),
                    nn.ReLU(),
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(env.get_action_space())),
                    nn.Softmax(dim=-1)
                )
        
        # critic
#        self.feature2 = nn.Sequential(
#                    nn.Conv2d(1,16,(3,3),1,1),
#                    nn.BatchNorm2d(16),
#                    nn.ReLU(),
#                    nn.MaxPool2d(2),
#                    nn.Conv2d(16,32,(3,3),1,1),
#                    nn.BatchNorm2d(32),
#                    nn.ReLU(),
#                    nn.MaxPool2d(2),
#                    nn.Conv2d(32,32,(3,3),1,1),
#                    nn.BatchNorm2d(32),
#                    nn.ReLU(),
#                    nn.MaxPool2d(2),
#                    nn.Flatten()
#                    )
#        self.reg2 = nn.Sequential(
#                    nn.Linear(3*3*32, 500),
#                    nn.ReLU(),
#                    nn.Linear(500, 256),
#                    nn.ReLU(),
#                    nn.Linear(256, 1)
#                )

        self.feature2 = nn.Sequential(
            nn.Conv2d(2, 16, (8, 8), 4, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.reg2 = nn.Sequential(
                    nn.Linear(2 * 2 * 32, 500),
                    nn.ReLU(),
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )

        self.train()
        
    def action_layer(self, x1):
        # Generating embedding vectors: Convert input [6,1, 25,25] to embedding [6, 500] (N, dim) 1-D embedding vectors for each agents
        x = self.embeding_layer(x1)

        num_agents= CONST.NUM_AGENTS

        self_in = x

        # check if single or batch
        if x.shape[0] == num_agents:
            self.graph.ndata['x'] = x
            # run the graph convolution (attention) to get new feature x and graph
            x = self.GAT(self.graph, self.graph.ndata['x'])
        # else doing batch
        else:
            batch_sz = x.shape[0] // num_agents

            G = dgl.batch([self.graph] * batch_sz)

            G.ndata['x'] = x
            x = self.GAT(G, G.ndata['x'])




        self.attention_mat= self.GAT.attention_mat

        # concatenate => z=  (h0 || h1) as the output from GAT
        x = torch.cat((self_in, x), 1)

        # get action distribution
        x = self.reg1(x)

        return x
    
    def value_layer(self, x1):
        x = self.feature2(x1)
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
                memory.actions.append(action[agent_index].view(1))
                memory.logprobs.append(dist.log_prob(action[agent_index])[agent_index])
                action_list.append(action[agent_index].item())
#                action_list.append(action[agent_index].view(1))
        return action_list
    
    def act_max(self, state, memory, num_agents):
#        with torch.no_grad():
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
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        att= self.attention_mat.view(-1, CONST.NUM_AGENTS)
        att_dist= Categorical(att)
        att_entropy = att_dist.entropy()

        
        action_logprobs = torch.diag(dist.log_prob(action))
#        action_logprobs = dist.log_prob(action)
        action_logprobs = action_logprobs.view(-1,1)
        dist_entropy = dist.entropy()

        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy, att_entropy
        
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









        
        self.MseLoss = nn.MSELoss()
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_CNN{random.randint(0, 1000)}")
        print(f"Log Dir: {self.sw.log_dir}")
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        all_rewards = []
        discounted_reward_list = [0]* int(CONST.NUM_AGENTS)
        agent_index_list = list(range(CONST.NUM_AGENTS)) * int(len(memory.rewards)/ CONST.NUM_AGENTS)
        for reward, is_terminal, agent_index in zip(reversed(memory.rewards), reversed(memory.is_terminals), reversed(agent_index_list)):
            if is_terminal:
                discounted_reward_list[agent_index] = 0
            discounted_reward_list[agent_index] = reward + (self.gamma * discounted_reward_list[agent_index])
            all_rewards.insert(0, discounted_reward_list[agent_index])
        
        # Normalizing the rewards:
#        all_rewards = torch.tensor(all_rewards).to(device)
#        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)

#        all_rewards = np.array(all_rewards)
#        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)

        all_rewards = torch.tensor(all_rewards).to(device)
        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        
        minibatch_sz = CONST.NUM_AGENTS * CONST.LEN_EPISODE
            
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
                rewards = mini_rewards #torch.from_numpy(mini_rewards).float().to(device)
                
                prev = i
                
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy, att_entropy = self.policy.evaluate(old_states, old_actions)
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs.view(-1,1) - old_logprobs.view(-1,1).detach())


                    
                # Finding Surrogate Loss:
                advantages = rewards - state_values.detach()
                advantages = advantages.view(-1,1)
    #            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean() + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy.mean() + 0.01 * att_entropy.mean()

                
                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return att_entropy.mean().item()
        
    def formatInput(self, states):
        out = []
        for i in range(len(states[2])):
            temp = [states[2][i],states[3][i]]
            out.append(temp)
        return np.array(out)
    
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, loss, rewardHistory, agent_RwdDict, lenEpisode):
        if loss:
            self.sw.add_scalar('6.att_entropy', loss, episode)
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
        
    def saveModel(self, filePath, per_save = False, episode =0):
        if per_save == False:
            torch.save(self.policy.state_dict(), f"{filePath}/{self.policy.__class__.__name__}.pt")
        else:
            torch.save(self.policy.state_dict(), f"{filePath}/{self.policy.__class__.__name__}_{episode}.pt")
    
    def loadModel(self, filePath, cpu = 0):

        if cpu == 1:
            self.policy.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
        else:
            self.policy.load_state_dict(torch.load(filePath))
        self.policy.eval()
