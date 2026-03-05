#!/usr/bin/env python3
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _learner import AttentionLearner
from MODEL.model import VECTRA
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

	marl = VECTRA(
            data.CUST_FEAT_SIZE,
		dyna.VEH_STATE_SIZE,
		model_size = args.model_size,
		layer_count = 3,
		head_count = 8,
		ff_size = 512,
		tanh_xplor = args.tanh_xplor,
		greedy = False,
		edge_feat_size = 8,
		cust_k = args.cust_k,
		memory_size = 128,
		lookahead_hidden = 128,
		dropout = args.dropout,
		adaptive_depth = False,
		adaptive_min_layers = args.adaptive_min_layers,
		adaptive_easy_ratio = args.adaptive_easy_ratio,
		latent_bottleneck = False,
		latent_tokens = args.latent_tokens,
		latent_min_nodes = args.latent_min_nodes,
		)
	with torch.no_grad():
		_ = marl(dyna)

	print("AttentionLearner params:", _count_params(attn))
	print("VECTRA params:", _count_params(marl))


if __name__ == "__main__":
	main()
