# rl_maxm-visibility
Using Reinforcement learning to solve the persistent monitoring problem using a Multi-Agent setup. A decentralized system using Proximal Policy Optimization (PPO) is trained with various scenarios. There is no cooperation introduced, just a local view of the agent (25x25 sq. units area around the agent) and compressed minimap (50x50 sq. units environment compressed to 25x25 sq. units) is provided as an input to the agent which will then decide to execute one of 4 descrete motions (stay, move up, left, down and right).

The environment is made of descrete element that accumulate a penalty value based on a pre-defined decay rate until the agent observes the element in it visibility region. The sum of all the penalty values of all the elements of the map is used to train the agent. The agent must uncover the right behavior to keep observing every descrete element in the map to achieve high reward (less penalty), hence Persistent Monitoring Problem.

# Link to Logs
[Logs to Discussions and work on the project](https://drive.google.com/open?id=1MSeY968ifq7bBtcaZ7DvFnWGinB2aJC3II-zEnzQXVM)

# Training Models
1. A single agent was trained on an environment with 2 compartments/ rooms. The final behavior can be seen bellow
![1Agent_2Room](https://github.com/amrish1222/rl_multi_agent/blob/4_room/results/1Agent_2Room.gif)

2. Two agents were trained on an environment with 2 compartments/ rooms. The final behavior can be seen bellow
![2Agent_2Room](https://github.com/amrish1222/rl_multi_agent/blob/4_room/results/2Agent_2Room.gif)

3. Two agents were trained on an environment with 2 compartments/ rooms. The final behavior can be seen bellow
![2Agent_4Room](https://github.com/amrish1222/rl_multi_agent/blob/4_room/results/2Agent_4Room.gif)

### This system is further trained to find the maximum number of agents it can handle before a cooperation based methodology is used.
