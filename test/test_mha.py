#!/usr/bin/env python3

from layers._mha import _MHA_V1, _MHA_V2 , MixedScore_MultiHeadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    import time
    from itertools import chain

    mha = _MHA_V1(8,4,128)
    mha_v2 = _MHA_V2(8,4,128)
    mha_v2.load_state_dict(mha.state_dict())
    proj = nn.Linear(128,1)

    q = torch.rand(512,1,4)
    k = torch.rand(512,10,128)
    v = torch.rand(512,10,128)
    m = torch.randint(0,2, (512,1,10), dtype = torch.bool)
    gt = torch.zeros(512,1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mha.to(dev)
    mha_v2.to(dev)
    proj.to(dev)
    q,k,v,m,gt = q.to(dev), k.to(dev), v.to(dev), m.to(dev), gt.to(dev)

    IT = 100

    ref = mha(q,k,v,m)
    x = proj(ref.squeeze(1))
    loss = F.smooth_l1_loss(x, gt)
    loss.backward()
    grad_ref = [p.grad for p in mha.parameters()]
    all_close_fw = 0
    all_close_bw = 0

    for i,mha_ver in enumerate([mha, mha_v2]):
        mu_t_fw = 0
        mu_t_bw = 0
        for it in range(IT):
            st_t = time.monotonic()
            o = mha_ver(q,k,v,m)
            mu_t_fw += (time.monotonic() - st_t) / IT
            if torch.allclose(o,ref, atol = 1e-9):
                all_close_fw += 1

            x = proj(o.squeeze(1))
            loss = F.smooth_l1_loss(x, gt)
            mha_ver.zero_grad()
            proj.zero_grad()
            st_t = time.monotonic()
            loss.backward()
            mu_t_bw += (time.monotonic() - st_t) / IT
            if all( torch.allclose(p.grad, gp_ref, atol = 1e-9) for p,gp_ref in zip(mha_ver.parameters(), grad_ref) ):
                all_close_bw += 1
        print("V{} : \t\t FW = {:.3f}ms match {:.0%} \t\t BW = {:.3f}ms match {:.0%}".format(i+1, mu_t_fw * 1000, all_close_fw / IT, mu_t_bw * 1000, all_close_bw / IT))
    
    print("\n" + "="*80)
    print("Testing MixedScore_MultiHeadAttention")
    print("="*80)
    
    # Initialize MixedScore MHA with same architecture
    ms_mha = MixedScore_MultiHeadAttention(
        head_count=8, 
        query_size=4, 
        key_size=128,
        ms_hidden_dim=16
    ).to(dev)
    
    # Create cost matrix (e.g., distance matrix in VRP)
    cost_mat = torch.rand(512, 1, 10).to(dev)  # N x L_q x L_kv
    
    print("\n1. Testing forward pass with cost_mat...")
    try:
        ms_out = ms_mha(q, k, v, mask=m, cost_mat=cost_mat)
        print(f"   ✓ Forward pass successful. Output shape: {ms_out.shape}")
        print(f"   ✓ Expected shape: {ref.shape}")
        assert ms_out.shape == ref.shape, f"Shape mismatch: {ms_out.shape} vs {ref.shape}"
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
    
    print("\n2. Testing backward pass...")
    try:
        x_ms = proj(ms_out.squeeze(1))
        loss_ms = F.smooth_l1_loss(x_ms, gt)
        ms_mha.zero_grad()
        proj.zero_grad()
        loss_ms.backward()
        print(f"   ✓ Backward pass successful. Loss: {loss_ms.item():.6f}")
        grad_ms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in ms_mha.parameters()]
        print(f"   ✓ Gradient norms: min={min(grad_ms):.6f}, max={max(grad_ms):.6f}, mean={sum(grad_ms)/len(grad_ms):.6f}")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
    
    print("\n3. Testing precompute functionality...")
    try:
        ms_mha.precompute(k, v)
        ms_out_precomp = ms_mha(q, cost_mat=cost_mat, mask=m)
        print(f"   ✓ Precompute successful. Output shape: {ms_out_precomp.shape}")
        
        # Check if precomputed output matches non-precomputed
        if torch.allclose(ms_out, ms_out_precomp, atol=1e-6):
            print(f"   ✓ Precomputed output matches non-precomputed (max diff: {(ms_out - ms_out_precomp).abs().max().item():.2e})")
        else:
            print(f"   ✗ Precomputed output differs (max diff: {(ms_out - ms_out_precomp).abs().max().item():.2e})")
    except Exception as e:
        print(f"   ✗ Precompute test failed: {e}")
    
    print("\n4. Testing self-attention mode...")
    try:
        # Initialize new MHA with matching dimensions for self-attention
        ms_mha_self = MixedScore_MultiHeadAttention(
            head_count=8, 
            query_size=128,  # Same as key_size for self-attention
            ms_hidden_dim=16
        ).to(dev)
        
        q_self = torch.rand(512, 10, 128).to(dev)
        m_self = torch.randint(0, 2, (512, 10), dtype=torch.bool).to(dev)
        cost_mat_self = torch.rand(512, 10, 10).to(dev)
        
        ms_out_self = ms_mha_self(q_self, mask=m_self, cost_mat=cost_mat_self)
        print(f"   ✓ Self-attention successful. Output shape: {ms_out_self.shape}")
        assert ms_out_self.shape == q_self.shape, f"Shape mismatch in self-attention"
    except Exception as e:
        print(f"   ✗ Self-attention test failed: {e}")
    
    print("\n5. Performance comparison (MixedScore vs V2)...")
    # Reinitialize for fair comparison
    ms_mha = MixedScore_MultiHeadAttention(8, 4, 128, ms_hidden_dim=16).to(dev)
    
    mu_t_fw_ms = 0
    mu_t_bw_ms = 0
    
    for it in range(IT):
        st_t = time.monotonic()
        o_ms = ms_mha(q, k, v, mask=m, cost_mat=cost_mat)
        mu_t_fw_ms += (time.monotonic() - st_t) / IT
        
        x_ms = proj(o_ms.squeeze(1))
        loss_ms = F.smooth_l1_loss(x_ms, gt)
        ms_mha.zero_grad()
        proj.zero_grad()
        st_t = time.monotonic()
        loss_ms.backward()
        mu_t_bw_ms += (time.monotonic() - st_t) / IT
    
    print(f"   V2 (standard):    FW = {mu_t_fw * 1000:.3f}ms, BW = {mu_t_bw * 1000:.3f}ms")
    print(f"   MixedScore:       FW = {mu_t_fw_ms * 1000:.3f}ms, BW = {mu_t_bw_ms * 1000:.3f}ms")
    print(f"   Overhead:         FW = {(mu_t_fw_ms/mu_t_fw - 1)*100:+.1f}%, BW = {(mu_t_bw_ms/mu_t_bw - 1)*100:+.1f}%")
    
    print("\n6. Testing cost_mat requirement...")
    try:
        ms_out_no_cost = ms_mha(q, k, v, mask=m)
        print(f"   ✗ Should have raised ValueError when cost_mat is None")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    print("\n7. Comparing outputs: MixedScore vs V2...")
    # Reinitialize để tránh NaN từ test trước
    ms_mha_compare = MixedScore_MultiHeadAttention(8, 4, 128, ms_hidden_dim=16).to(dev)
    
    # Test với cost_mat = constant (không ảnh hưởng)
    cost_mat_const = torch.zeros(512, 1, 10).to(dev)
    ms_out_const = ms_mha_compare(q, k, v, mask=m, cost_mat=cost_mat_const)
    
    # Test với cost_mat = varied (ảnh hưởng lớn)
    cost_mat_varied = torch.randn(512, 1, 10).to(dev) * 10.0
    ms_out_varied = ms_mha_compare(q, k, v, mask=m, cost_mat=cost_mat_varied)
    
    # Test với cost_mat random (như test trước)
    cost_mat_random = torch.rand(512, 1, 10).to(dev)
    ms_out_random = ms_mha_compare(q, k, v, mask=m, cost_mat=cost_mat_random)
    
    # So sánh với V2
    v2_out = mha_v2(q, k, v, m)
    
    diff_const = (ms_out_const - v2_out).abs()
    diff_varied = (ms_out_varied - v2_out).abs()
    diff_random = (ms_out_random - v2_out).abs()
    
    print(f"   • cost_mat=zeros:  max_diff={diff_const.max().item():.6f}, mean_diff={diff_const.mean().item():.6f}")
    print(f"   • cost_mat=random: max_diff={diff_random.max().item():.6f}, mean_diff={diff_random.mean().item():.6f}")
    print(f"   • cost_mat=varied: max_diff={diff_varied.max().item():.6f}, mean_diff={diff_varied.mean().item():.6f}")
    
    # Kiểm tra correlation
    ms_flat = ms_out_random.flatten()
    v2_flat = v2_out.flatten()
    correlation = torch.corrcoef(torch.stack([ms_flat, v2_flat]))[0, 1].item()
    print(f"   • Correlation(MixedScore, V2) với cost_mat=random: {correlation:.4f}")
    
    print("\n   Interpretation:")
    print("   - Khác biệt nhỏ: cost_mat ít ảnh hưởng (MLP weights gần 0 cho cost channel)")
    print("   - Khác biệt lớn: cost_mat ảnh hưởng mạnh (MLP đã học kết hợp 2 signals)")
    print("   - Correlation cao (>0.9): cả 2 attention patterns tương đồng")
    print("   - Correlation thấp (<0.7): cost_mat thay đổi hoàn toàn attention pattern")
    
    print("\n" + "="*80)
    print("MixedScore_MultiHeadAttention tests completed!")
    print("="*80)
