#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MODEL.model import EdgeEnhencedLearner
from problems import VRP_Dataset, VRP_Environment, VRPTW_Dataset, VRPTW_Environment
from layers import reinforce_loss


def _run_vrp():
	data = VRP_Dataset.generate(4, 10, 2)
	dyna = VRP_Environment(data)

	learner = EdgeEnhencedLearner(data.CUST_FEAT_SIZE, dyna.VEH_STATE_SIZE, tanh_xplor = 10)

	actions, logps, rewards = learner(dyna)
	print("Forward pass ok for VRP")

	loss = reinforce_loss(logps, rewards)
	loss.backward()
	print("Backward pass ok for VRP")


def _run_vrptw():
	data = VRPTW_Dataset.generate(4, 10, 2)
	dyna = VRPTW_Environment(data)

	learner = EdgeEnhencedLearner(data.CUST_FEAT_SIZE, dyna.VEH_STATE_SIZE, tanh_xplor = 10)

	actions, logps, rewards = learner(dyna)
	print("Forward pass ok for VRPTW")

	loss = reinforce_loss(logps, rewards)
	loss.backward()
	print("Backward pass ok for VRPTW")


def main():
	_run_vrp()
	_run_vrptw()


if __name__ == "__main__":
	main()
