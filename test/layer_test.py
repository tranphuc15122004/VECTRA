import sys
import os
import torch

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import (
	GraphEncoder,
	FleetEncoder,
	CrossEdgeFusion,
	CoordinationMemory,
	OwnershipHead,
	LookaheadHead,
	EdgeFeatureEncoder,
)


def _assert_finite(tensor, name):
	assert torch.isfinite(tensor).all(), f"{name} has non-finite values"


def test_graph_encoder_shapes_and_masking():
	torch.manual_seed(0)
	batch, length, model_size = 2, 5, 32
	head_count = 8
	ff_size = 64
	inputs = torch.randn(batch, length, model_size)
	coords = torch.rand(batch, length, 2)
	mask = torch.zeros(batch, length, dtype=torch.bool)
	mask[:, -1] = True

	enc = GraphEncoder(layer_count=2, head_count=head_count, model_size=model_size, ff_size=ff_size)
	out = enc(inputs, mask=mask, coords=coords)
	assert out.shape == (batch, length, model_size)
	_assert_finite(out, "GraphEncoder output")
	assert torch.all(out[mask] == 0)


def test_graph_encoder_cost_mat_override():
	torch.manual_seed(0)
	batch, length, model_size = 2, 5, 32
	enc = GraphEncoder(layer_count=1, head_count=8, model_size=model_size, ff_size=64)
	inputs = torch.randn(batch, length, model_size)
	cost_mat = torch.rand(batch, length, length)
	out = enc(inputs, cost_mat=cost_mat)
	assert out.shape == (batch, length, model_size)
	_assert_finite(out, "GraphEncoder output (cost_mat)")


def test_fleet_encoder_shapes_and_knn_mask():
	torch.manual_seed(0)
	batch, veh_count, veh_state = 2, 4, 6
	model_size = 32
	head_count = 8
	ff_size = 64

	vehicles = torch.randn(batch, veh_count, veh_state)
	dist = torch.cdist(vehicles[:, :, :2], vehicles[:, :, :2])
	time_gap = torch.abs(vehicles[:, :, 3:4] - vehicles[:, :, 3:4].transpose(1, 2))
	capa_gap = torch.abs(vehicles[:, :, 2:3] - vehicles[:, :, 2:3].transpose(1, 2))
	fleet_edges = torch.stack((dist, time_gap.squeeze(-1), capa_gap.squeeze(-1)), dim=-1)

	enc = FleetEncoder(layer_count=2, head_count=head_count, model_size=model_size, ff_size=ff_size, k=3)
	out = enc(vehicles, fleet_edges)
	assert out.shape == (batch, veh_count, model_size)
	_assert_finite(out, "FleetEncoder output")


def test_edge_feature_and_fusion_shapes():
	torch.manual_seed(0)
	batch, length, model_size = 2, 6, 32
	head_count = 8
	edge_feat_size = 8

	edge_feat = torch.randn(batch, 1, length, edge_feat_size)
	edge_encoder = EdgeFeatureEncoder(edge_feat_size=edge_feat_size, model_size=model_size)
	edge_emb = edge_encoder(edge_feat)
	assert edge_emb.shape == (batch, 1, length, model_size)
	_assert_finite(edge_emb, "EdgeFeatureEncoder output")

	veh_repr = torch.randn(batch, 1, model_size)
	cust_repr = torch.randn(batch, length, model_size)
	fusion = CrossEdgeFusion(head_count=head_count, model_size=model_size)
	score = fusion(veh_repr, cust_repr, edge_emb)
	assert score.shape == (batch, 1, length)
	_assert_finite(score, "CrossEdgeFusion output")


def test_edge_feature_encoder_gradients():
	torch.manual_seed(0)
	batch, length, model_size = 2, 6, 32
	edge_feat_size = 8
	edge_feat = torch.randn(batch, 1, length, edge_feat_size, requires_grad=True)
	edge_encoder = EdgeFeatureEncoder(edge_feat_size=edge_feat_size, model_size=model_size)
	edge_emb = edge_encoder(edge_feat)
	loss = edge_emb.mean()
	loss.backward()
	assert edge_feat.grad is not None
	_assert_finite(edge_feat.grad, "EdgeFeatureEncoder grad")


def test_coordination_memory_update():
	torch.manual_seed(0)
	batch, veh_count, model_size = 2, 4, 32
	hidden_size = 16

	memory = torch.zeros(batch, veh_count, hidden_size)
	veh_idx = torch.zeros(batch, 1, dtype=torch.long)
	veh_repr = torch.randn(batch, 1, model_size)
	cust_repr = torch.randn(batch, 1, model_size)
	edge_emb = torch.randn(batch, 1, 1, model_size)

	mem = CoordinationMemory(veh_state_size=model_size, hidden_size=hidden_size)
	updated = mem.update(memory, veh_idx, veh_repr, cust_repr, edge_emb)
	assert updated.shape == memory.shape
	_assert_finite(updated, "CoordinationMemory output")
	# Ensure only one vehicle index changed
	changed = (updated - memory).abs().sum(dim=-1)
	assert torch.all(changed[:, 1:] == 0)


def test_ownership_and_lookahead_heads():
	torch.manual_seed(0)
	batch, veh_count, cust_count, model_size = 2, 3, 5, 32
	hidden_size = 16

	veh_memory = torch.randn(batch, veh_count, hidden_size)
	cust_repr = torch.randn(batch, cust_count, model_size)
	owner = OwnershipHead(model_size=model_size)
	owner_logits = owner(veh_memory, cust_repr)
	assert owner_logits.shape == (batch, veh_count, cust_count)
	_assert_finite(owner_logits, "OwnershipHead output")

	veh_repr = torch.randn(batch, 1, model_size)
	edge_emb = torch.randn(batch, 1, cust_count, model_size)
	lookahead = LookaheadHead(model_size=model_size, hidden_size=64)
	la = lookahead(veh_repr, cust_repr, edge_emb)
	assert la.shape == (batch, 1, cust_count)
	_assert_finite(la, "LookaheadHead output")


def test_device_consistency_cpu_gpu():
	if not torch.cuda.is_available():
		return
	torch.manual_seed(0)
	batch, length, model_size = 2, 5, 32
	enc = GraphEncoder(layer_count=1, head_count=8, model_size=model_size, ff_size=64)
	inputs = torch.randn(batch, length, model_size)
	coords = torch.rand(batch, length, 2)
	out_cpu = enc(inputs, coords=coords)
	enc_cuda = enc.cuda()
	out_gpu = enc_cuda(inputs.cuda(), coords=coords.cuda()).cpu()
	assert out_cpu.shape == out_gpu.shape
	_assert_finite(out_gpu, "GraphEncoder GPU output")


def test_head_divisibility_errors():
	try:
		_ = GraphEncoder(layer_count=1, head_count=3, model_size=32, ff_size=64)
		excepted = False
	except ValueError:
		excepted = True
	assert excepted

	try:
		_ = CrossEdgeFusion(head_count=3, model_size=32)
		excepted = False
	except ValueError:
		excepted = True
	assert excepted


def main():
	tests = [
		test_graph_encoder_shapes_and_masking,
		test_graph_encoder_cost_mat_override,
		test_fleet_encoder_shapes_and_knn_mask,
		test_edge_feature_and_fusion_shapes,
		test_edge_feature_encoder_gradients,
		test_coordination_memory_update,
		test_ownership_and_lookahead_heads,
		test_device_consistency_cpu_gpu,
		test_head_divisibility_errors,
	]
	for test in tests:
		try:
			test()
			print(f"[OK] {test.__name__}")
		except AssertionError as exc:
			print(f"[FAIL] {test.__name__}: {exc}")
			sys.exit(1)
	print("All layer tests passed.")


if __name__ == "__main__":
	main()
