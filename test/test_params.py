#!/usr/bin/env python3
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _learner import AttentionLearner
from MODEL.model import EdgeEnhencedLearner
from utils import parse_args
from problems import VRP_Dataset, VRP_Environment


def _count_params(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
	args = parse_args([])

	data = VRP_Dataset.generate(1, args.customers_count, args.vehicles_count)
	dyna = VRP_Environment(data)

	attn = AttentionLearner(
		data.CUST_FEAT_SIZE,
		dyna.VEH_STATE_SIZE,
		args.model_size,
		args.layer_count,
		args.head_count,
		args.ff_size,
		args.tanh_xplor,
	)

	marl = EdgeEnhencedLearner(
		data.CUST_FEAT_SIZE,
		dyna.VEH_STATE_SIZE,
		128,
		2,
		4,
		256,
		args.tanh_xplor,
		False,
		args.edge_feat_size,
		5,
		args.memory_size,
		args.lookahead_hidden,
	)

	with torch.no_grad():
		_ = marl(dyna)

	print("AttentionLearner params:", _count_params(attn))
	print("EdgeEnhencedLearner params:", _count_params(marl))


if __name__ == "__main__":
	main()
