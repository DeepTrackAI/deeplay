import torch
import torch.nn as nn

x = torch.randn(2, 2)
print("sending x to mps")
x = x.to("mps")
