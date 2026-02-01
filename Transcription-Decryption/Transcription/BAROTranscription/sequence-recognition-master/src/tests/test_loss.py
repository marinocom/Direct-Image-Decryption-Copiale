from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


output = torch.tensor([[-5, 5.05, -0.3]])
target = torch.tensor([1])


#%%


label_smoothing = 0.1
gamma = 1.0
ignore_index = None
reduction = "mean"


weights = None

# sigmoid = nn.Sigmoid()
sigmoid = nn.Softmax(dim=-1)

positions = F.one_hot(target, num_classes=output.size(-1))
mask_correct = positions > 0

probs = sigmoid(output)

print(probs)

#%%

if label_smoothing > 0.0:
    nclasses = output.shape[-1]
    probs = ((1 - label_smoothing) * probs) + (
        label_smoothing / nclasses
    )
print(probs)

#%%

scores = torch.where(mask_correct, probs, 1 - probs)
print(scores)

#%%

loss = -((1 - scores) ** gamma) * scores.log()
print(loss)

#%%


if weights is not None:
    scores = scores * weights[None, :].expand(scores.size(0), -1)

if ignore_index is not None:
    scores = torch.where(
        (target != ignore_index)[:, None].expand(-1, output.shape[-1]),
        scores,
        torch.tensor(0.0),
    )

if reduction == "mean":
    loss = loss.flatten().mean()
else:
    loss = loss.flatten().sum()

print(loss)
