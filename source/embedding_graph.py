import dgl
import numpy as np
import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
#import gat_mod
#from gat_mod import GATConv
from dgl.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class embedding_layer(nn.Module):
    def __init__(self):
        super(embedding_layer, self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Conv2d(2,16,(3,3),1,1),
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

        self.layer2 = nn.Sequential(
            nn.Linear(3 * 3 * 32, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
        )

        self.train()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


"""

use average to combine multi-head attetion results
=> edge (E, H, 1)
E is number of edges
H is number of heads 
1 dimension is softmaxed alpha score 

"""


def ave_heads(h, num_h):
    res = 0.0
    for i in range(num_h):
        res += h[:, i, :]

    res = res / ((float)(num_h))

    return res


class GAT(nn.Module):
    def __init__(self, in_feats, heads):
        super(GAT, self).__init__()

        self.heads = heads

        self.conv1 = GATConv(in_feats, in_feats, heads)
        #self.conv2 = GATConv(in_feats, in_feats, heads)

    def forward(self, g, inputs):
        # print("Detail of convolution result for each layer:")

        l1 = self.conv1(g, inputs)
        # print("multiple heads before merge")
        # print((g.edata['a']))
        # print(l1)
        l1 = ave_heads(l1, self.heads)
        # print(g.edata['a'])
        # print(f"The result after the 1st Conv layer: \n {l1}")

        # print((g.edata))

        # l2, g = self.conv2(g, l1)
        # print("multiple heads before merge")
        # print((g.edata['a']))
        # print(l2)
        # g.edata['a'] = ave_heads(g.edata['a'], self.heads)
        # l2 = ave_heads(l2, self.heads)
        # print(g.edata['a'])
        # print(f"The result after the 2nd Conv layer: \n {l2}")
        # print((g.edata))
        # combining to get final features for all nodes (+)
        return l1


"""
generate Fully connected graph for n agents: (with self loop: node i -> node i)
"""


def FC_graph(num_nodes):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    fc_src = [i for i in range(num_nodes)]
    for i in range(num_nodes):
        node_i = [i] * num_nodes
        g.add_edges(fc_src, node_i)

    return g



"""
G = FC_graph(5)

"""

"""
show the graph structure
"""

"""
# display graph info


print("Graph information: ")
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())
print()
print()

pos = nx.kamada_kawai_layout(G.to_networkx())
nx.draw(G.to_networkx(), pos, with_labels=True, node_color=[[.7, .7, .7]])
plt.show()

"""

"""
image_batch: size (N, W, H)
N number of image, W width, H height  

"""

"""
# for testing 
def embeddinhg_graph(image_batch, G):
    embed = embedding_layer().to(device)
    input = torch.tensor([]).to(device)

    for i in range(image_batch.shape[0]):
        image = image_batch[i].unsqueeze(0).unsqueeze(0)
        input = torch.cat((input, embed(image)), 0)

    input = input.to(device)

    G.ndata['x'] = input
    net = GAT(G.ndata['x'].shape[1], 3)
    res, G = net(G, G.ndata['x'])

    return res, G

"""


"""
get attention matrix (all incoming edges weights for all nodes)
"""


def get_att_matrix(graph):
    num = graph.number_of_nodes()
    att_martix = torch.zeros((num, num))
    for i in range(num):
        att_martix[i] = graph.edata['a'][graph.in_edges(i, 'eid')].detach().view(1, -1)[0]

    return att_martix


"""
test part:
"""

""""
input = torch.ones(25, 25).to(device)

input = input.unsqueeze(0)

input = torch.cat((input, input * 2, input * 3, input * 4, input * 5), 0)

res, G = embeddinhg_graph(input, G)

print("final features for each node ")
print(res)
print()
print()

print("final attention matrix for each node")
print(get_att_matrix(G))
print()
print()

"""



