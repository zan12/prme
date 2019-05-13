import numpy as np
import torch

from argument import parse_arguments
from prme import PRME

# parse arguments
args = parse_arguments()

# load data
data = np.load(args.data_filename)

# train model
# fix random seeds for reproducing results
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

model = PRME(args)
model.fit(data)