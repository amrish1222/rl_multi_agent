import torch
from torch.distributions import Categorical

p_tensor = torch.Tensor([0.25, 0.25, 0.25, 0.25,0.7, 0.1, 0.1, 0.1,0.25, 0.25, 0.25, 0.25,0.7, 0.1, 0.1, 0.1])
print(p_tensor.shape)
p_tensor= p_tensor.view(-1,4)
print(p_tensor)
entropy2 = Categorical(probs = p_tensor).entropy()
print(entropy2.shape)
print(entropy2)


p_tensor = torch.Tensor([0.7, 0.1, 0.1, 0.1])
entropy2 = Categorical(probs = p_tensor).entropy()
print(entropy2)