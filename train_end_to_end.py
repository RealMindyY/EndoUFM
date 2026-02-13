from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import random
from trainer_end_to_end_sam import Trainer
from options import MonodepthOptions
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
options = MonodepthOptions()
opts = options.parse()

def random_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    random_seeds(314)
    trainer = Trainer(opts)
    trainer.train()
