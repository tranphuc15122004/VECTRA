#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MODEL.model import EdgeEnhencedLearner
from problems import VRP_Dataset, VRP_Environment
from baselines import *
from layers import reinforce_loss

data = VRP_Dataset.generate(6, 10, 2)
dyna = VRP_Environment(data)
dyna.reset()

learner = EdgeEnhencedLearner(data.CUST_FEAT_SIZE, dyna.VEH_STATE_SIZE, tanh_xplor = 10)

bl = NearestNeighbourBaseline(learner)

a,l,r,b = bl(dyna)
loss = reinforce_loss(l,r,b)
print("Loss with near_nb baseline:", loss)

#################

bl2 = RolloutBaseline(learner, 3)
a,l,r,b = bl2(dyna)
loss = reinforce_loss(l,r,b)
print("Loss with rollout baseline:", loss)

#################

bl3 = CriticBaseline(learner, 10)

a,l,r,b = bl3(dyna)
loss = reinforce_loss(l,r,b)
print("Loss with critic baseline:", loss)
