import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import GraphEncoder, FleetEncoder


def _assert_finite(tensor, name):
    assert torch.isfinite(tensor).all(), f"{name} has non-finite values"


def test_graph_encoder_adaptive_depth_shapes():
    torch.manual_seed(0)
    batch, length, model_size = 3, 10, 32
    enc = GraphEncoder(
        layer_count=3,
        head_count=8,
        model_size=model_size,
        ff_size=64,
        k=5,
        adaptive_depth=True,
        min_layers=1,
        easy_ratio=0.7,
    )

    inputs = torch.randn(batch, length, model_size)
    coords = torch.rand(batch, length, 2)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[0, -2:] = True

    out = enc(inputs, mask=mask, coords=coords)
    assert out.shape == (batch, length, model_size)
    _assert_finite(out, "GraphEncoder adaptive output")
    assert torch.all(out[mask] == 0)


def test_graph_encoder_all_masked_stability():
    torch.manual_seed(0)
    batch, length, model_size = 2, 6, 32
    enc = GraphEncoder(
        layer_count=2,
        head_count=8,
        model_size=model_size,
        ff_size=64,
        k=4,
        adaptive_depth=True,
        min_layers=1,
        easy_ratio=0.7,
    )

    inputs = torch.randn(batch, length, model_size)
    coords = torch.rand(batch, length, 2)
    mask = torch.ones(batch, length, dtype=torch.bool)

    out = enc(inputs, mask=mask, coords=coords)
    assert out.shape == (batch, length, model_size)
    _assert_finite(out, "GraphEncoder all-masked output")
    assert torch.all(out == 0)


def test_fleet_encoder_adaptive_depth_3d_mask():
    torch.manual_seed(0)
    batch, veh_count, cust_count = 2, 4, 8
    veh_state, model_size = 6, 32

    enc = FleetEncoder(
        layer_count=3,
        head_count=8,
        model_size=model_size,
        ff_size=64,
        adaptive_depth=True,
        min_layers=1,
        easy_ratio=0.7,
    )

    vehicles = torch.randn(batch, veh_count, veh_state)
    cust_repr = torch.randn(batch, cust_count, model_size)
    mask = torch.zeros(batch, veh_count, cust_count, dtype=torch.bool)
    mask[:, 0, :] = True

    out = enc(vehicles, cust_repr, mask=mask)
    assert out.shape == (batch, veh_count, model_size)
    _assert_finite(out, "FleetEncoder adaptive output")
    assert torch.all(out[:, 0, :] == 0)


def test_fleet_encoder_2d_mask_support():
    torch.manual_seed(0)
    batch, veh_count, cust_count = 2, 3, 7
    veh_state, model_size = 5, 32

    enc = FleetEncoder(
        layer_count=2,
        head_count=8,
        model_size=model_size,
        ff_size=64,
        adaptive_depth=True,
        min_layers=1,
        easy_ratio=0.6,
    )

    vehicles = torch.randn(batch, veh_count, veh_state)
    cust_repr = torch.randn(batch, cust_count, model_size)
    mask = torch.zeros(batch, cust_count, dtype=torch.bool)
    mask[:, -1] = True

    out = enc(vehicles, cust_repr, mask=mask)
    assert out.shape == (batch, veh_count, model_size)
    _assert_finite(out, "FleetEncoder 2D-mask output")


def main():
    tests = [
        test_graph_encoder_adaptive_depth_shapes,
        test_graph_encoder_all_masked_stability,
        test_fleet_encoder_adaptive_depth_3d_mask,
        test_fleet_encoder_2d_mask_support,
    ]
    for test in tests:
        try:
            test()
            print(f"[OK] {test.__name__}")
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
            sys.exit(1)
    print("All SBG layer tests passed.")


if __name__ == "__main__":
    main()
