import numpy as np
import torch
import time

def kmeans_l1(x_bow, K):
  N, D = x_bow.shape
  x_bow_normalized = x_bow/torch.sum(x_bow, dim=1, keepdim=True).repeat(1,D)
  centers = x_bow_normalized[np.random.choice(N, K, replace=False), :]
  for ite in range(20):
    min_idx = torch.zeros(N)
    for n in range(N):
      min_idx[n] = torch.argmin(torch.sum((
          centers-x_bow[n:(n+1),:].repeat(K,1)).abs(), dim=1))
    for k in range(K):
      if torch.sum(min_idx==k)>0:
        centers[k,:] = torch.sum(
            x_bow_normalized[torch.nonzero(
            min_idx==k),:], dim=0)/torch.sum(min_idx==k).float().cuda()
      else:
        centers[k,:] = x_bow_normalized[np.random.choice(N, 1), :]
  centers_rank = np.argsort(torch.histc(min_idx, 
                                    bins=K, min=0, max=K-1).numpy())[::-1]
  return centers[centers_rank.copy(),:]