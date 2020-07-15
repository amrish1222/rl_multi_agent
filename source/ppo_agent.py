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
import time
import dgl

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


        self.bacth_x = None


        # actor

        # added embedding layer (shared)
        self.embeding_layer = EMG.embedding_layer()
        # added fully connected graph generator
        self.graph = EMG.FC_graph(num_agents)
        # added GAT network: with #attention head = 3
        self.GAT = EMG.GAT(250, 3)




        self.reg1 = nn.Sequential(
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(env.get_action_space())),
                    nn.Softmax(dim=-1)
                )
        
        # critic


        self.reg2 = nn.Sequential(
                    nn.Linear(500, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )

        self.train()
        
    def action_layer(self, x1):


        # Generating embedding vectors: Convert input [6,1, 25,25] to embedding [6, 500] (N, dim) 1-D embedding vectors for each agents
        x= self.embeding_layer(x1)


        self_in = x



        # check if single or batch
        if x.shape[0] == num_agents:
            self.graph.ndata['x'] = x
            # run the graph convolution (attention) to get new feature x and graph
            x, self.graph = self.GAT(self.graph, self.graph.ndata['x'])
        # else doing batch
        else:
            batch_sz= x.shape[0] // num_agents



            G = dgl.batch([self.graph] * batch_sz)
            G.ndata['x']= x
            x, _ = self.GAT(G, G.ndata['x'])



        # concatenate => z=  (h0 || h1) as the output from GAT
        x = torch.cat((self_in, x), 1)




        self.bacth_x = x




        # get action distribution
        x = self.reg1(x)

        return x
    
    def value_layer(self, x1):




        # check if single or batch
        if x1.shape[0] == num_agents:
            # Generating embedding vectors: Convert input [6,1, 25,25] to embedding [6, 500] (N, dim) 1-D embedding vectors for each agents
            x = self.embeding_layer(x1)

            self_in = x
            self.graph.ndata['x'] = x
            # run the graph convolution (attention) to get new feature x and graph
            x, self.graph = self.GAT(self.graph, self.graph.ndata['x'])
            # concatenate => z=  (h0 || h1) as the output from GAT
            x = torch.cat((self_in, x), 1)

        # else doing batch
        else:
            #directly use batch result from actor (shared struture)
            x= self.bacth_x



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
        
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

#        action_logprobs = torch.diag(dist.log_prob(action))
        action_logprobs = dist.log_prob(action)
        action_logprobs = action_logprobs.view(-1,1)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

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
                #a = time.time()
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                #print("evaluate: ", round(1000*(time.time() - a),2))
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
        out = []
        for i in range(len(states[2])):
            temp = [states[2][i], states[3][i]]
            out.append(temp)
        return np.array(out)
    
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
            title = '4. Agent ' + str(item + 1)
            if len(agent_RwdDict[item]) >= 100:
                avg_agent_rwd = agent_RwdDict[item][-100:]
            else:
                avg_agent_rwd = agent_RwdDict[item]
            avg_agent_rwd = mean(avg_agent_rwd)

            self.sw.add_scalar(title, avg_agent_rwd, len(agent_RwdDict[item]) - 1)


    def summaryWriter_close(self):
        self.sw.close()
        
    def saveModel(self, filePath):
        torch.save(self.policy, f"{filePath}/{self.policy.__class__.__name__}.pt")
    
    def loadModel(self, filePath):
        self.model = torch.load(filePath)
        self.model.eval()
