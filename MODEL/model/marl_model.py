from layers import (
    GraphEncoder,
    FleetEncoder,
    CrossEdgeFusion,
    CoordinationMemory,
    OwnershipHead,
    LookaheadHead,
    EdgeFeatureEncoder,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time


class ForwardProfiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.timings = defaultdict(float)

    def add(self, section, duration):
        self.timings[section] += duration

    def get_summary(self):
        total = sum(self.timings.values())
        if total == 0:
            return [], 0.0
        ordered = sorted(self.timings.items(), key=lambda item: -item[1])
        summary = [
            (name, duration, duration / total * 100.0)
            for name, duration in ordered
        ]
        return summary, total

class EdgeEnhencedLearner(nn.Module):
    def __init__(self, cust_feat_size, veh_state_size, model_size = 128,
            layer_count = 3, head_count = 8, ff_size = 512, tanh_xplor = 10, greedy = False,
            edge_feat_size = 8, cust_k = 15, memory_size = None, lookahead_hidden = 128 , dropout = 0.1,
            sbg_enable = False, sbg_cand_k = 0, sbg_adaptive_k = False,
            sbg_k_min = 8, sbg_k_max = None, sbg_late_penalty = 2.0,
            sbg_slack_weight = 0.5, sbg_owner_weight = 0.5,
            sbg_moe_enable = False, sbg_moe_strength = 0.15,
            sbg_moe_uncertainty = True, sbg_moe_min_strength = 0.01,
            sbg_moe_entropy_floor = 0.35, sbg_moe_margin_ceil = 1.5,
            sbg_moe_load_balance_coef = 0.01,
            adaptive_depth = False, adaptive_min_layers = 1, adaptive_easy_ratio = 0.6,
            latent_bottleneck = False, latent_tokens = 32, latent_min_nodes = 64):
        r"""
        :param model_size:  Dimension :math:`D` shared by all intermediate layers
        :param layer_count: Number of layers in customers' (graph) Transformer Encoder
        :param head_count:  Number of heads in all Multi-Head Attention layers
        :param ff_size:     Dimension of the feed-forward sublayers in Transformer Encoder
        :param tanh_xplor:  Enable tanh exploration and set its amplitude
        """
        super().__init__()

        self.model_size = model_size
        self.inv_sqrt_d = model_size ** -0.5

        self.tanh_xplor = tanh_xplor

        self.depot_embedding = nn.Linear(cust_feat_size, model_size)
        self.cust_embedding  = nn.Linear(cust_feat_size, model_size)
        self.cust_encoder    = GraphEncoder(
            layer_count,
            head_count,
            model_size,
            ff_size,
            k = cust_k,
            adaptive_depth = adaptive_depth,
            min_layers = adaptive_min_layers,
            easy_ratio = adaptive_easy_ratio,
        )

        self.fleet_encoder   = FleetEncoder(
            layer_count,
            head_count,
            model_size,
            ff_size,
            adaptive_depth = adaptive_depth,
            min_layers = adaptive_min_layers,
            easy_ratio = adaptive_easy_ratio,
        )
        self.edge_encoder    = EdgeFeatureEncoder(edge_feat_size, model_size)
        self.cross_fusion    = CrossEdgeFusion(head_count, model_size)

        self.coord_memory    = CoordinationMemory(veh_state_size, model_size if memory_size is None else memory_size)
        self.owner_head      = OwnershipHead(model_size)
        self.lookahead_head  = LookaheadHead(model_size, hidden_size = lookahead_hidden)

        self.cust_project    = nn.Linear(model_size, model_size)
        self.veh_project     = nn.Linear(model_size, model_size)

        # Advanced score fusion MLP to capture complex interactions between 
        # attention, vehicle-customer ownership/coordination, and lookahead scores.
        self.score_fusion = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Learnable MoE experts and router (registered only when MoE is enabled)
        if sbg_moe_enable:
            self.expert_local_net  = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
            self.expert_future_net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
            self.moe_router        = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))

        self.dropout         = nn.Dropout(dropout)
        self.greedy = greedy
        self.sbg_enable = sbg_enable
        self.sbg_cand_k = sbg_cand_k
        self.sbg_adaptive_k = sbg_adaptive_k
        self.sbg_k_min = sbg_k_min
        self.sbg_k_max = sbg_k_max
        self.sbg_late_penalty = sbg_late_penalty
        self.sbg_slack_weight = sbg_slack_weight
        self.sbg_owner_weight = sbg_owner_weight
        self.sbg_moe_enable = sbg_moe_enable
        self.sbg_moe_strength = sbg_moe_strength
        self.sbg_moe_uncertainty = sbg_moe_uncertainty
        self.sbg_moe_min_strength = sbg_moe_min_strength
        self.sbg_moe_entropy_floor = sbg_moe_entropy_floor
        self.sbg_moe_margin_ceil = sbg_moe_margin_ceil
        self.sbg_moe_load_balance_coef = sbg_moe_load_balance_coef
        self._moe_aux_loss = None
        self.adaptive_depth = adaptive_depth
        self.adaptive_min_layers = adaptive_min_layers
        self.adaptive_easy_ratio = adaptive_easy_ratio
        self.latent_bottleneck = latent_bottleneck
        self.latent_tokens = latent_tokens
        self.latent_min_nodes = latent_min_nodes
        self._forward_profiler = ForwardProfiler()


    def _build_bottleneck_indices(self, length, max_tokens, device):
        tokens = max(2, min(max_tokens, length))
        if tokens >= length:
            return torch.arange(length, device = device)
        idx = torch.linspace(1, length - 1, steps = tokens - 1, device = device)
        idx = idx.round().long().unique(sorted = True)
        if idx.numel() < tokens - 1:
            pad = torch.arange(1, length, device = device)
            mask = torch.ones_like(pad, dtype = torch.bool)
            mask[idx.clamp(max = length - 1)] = False
            extra = pad[mask][: (tokens - 1 - idx.numel())]
            idx = torch.cat((idx, extra), dim = 0)
            idx = idx.sort().values
        idx = idx[:tokens - 1]
        return torch.cat((torch.zeros(1, device = device, dtype = torch.long), idx), dim = 0)


    def _encode_customers_bottleneck(self, cust_emb, customers, mask):
        idx = self._build_bottleneck_indices(customers.size(1), self.latent_tokens, customers.device)
        idx_exp_d = idx[None, :, None].expand(customers.size(0), -1, cust_emb.size(-1))
        idx_exp_c = idx[None, :, None].expand(customers.size(0), -1, 2)

        reduced_emb = cust_emb.gather(1, idx_exp_d)
        reduced_coords = customers[:, :, :2].gather(1, idx_exp_c)
        reduced_mask = None
        if mask is not None:
            reduced_mask = mask.gather(1, idx[None, :].expand(mask.size(0), -1))

        reduced_enc = self.cust_encoder(reduced_emb, reduced_mask, coords = reduced_coords)

        full_coords = customers[:, :, :2]
        dist = torch.cdist(full_coords, reduced_coords)
        if reduced_mask is not None:
            dist = dist.masked_fill(reduced_mask[:, None, :], 1e9)
            all_masked = reduced_mask.all(dim = 1)
            if all_masked.any():
                dist = dist.clone()
                dist[all_masked, :, 0] = 0.0
        nearest = dist.argmin(dim = -1)
        full_enc = reduced_enc.gather(1, nearest[:, :, None].expand(-1, -1, reduced_enc.size(-1)))
        if mask is not None:
            full_enc = full_enc.masked_fill(mask.unsqueeze(-1), 0.0)
        return full_enc


    def _encode_customers(self, customers, mask = None):
        r"""
        :param customers: :math:`N \times L_c \times D_c` tensor containing minibatch of customers' features
        :param mask:      :math:`N \times L_c` tensor containing minibatch of masks
                where :math:`m_{nj} = 1` if customer :math:`j` in sample :math:`n` is hidden (pad or dyn), 0 otherwise
        """
        start = time.perf_counter()
        cust_emb = torch.cat((
            self.depot_embedding(customers[:,0:1,:]),
            self.cust_embedding(customers[:,1:,:])
            ), dim = 1) #.size() = N x L_c x D
        if mask is not None:
            cust_emb[mask] = 0
        use_bottleneck = (
            self.latent_bottleneck
            and self.latent_tokens is not None
            and self.latent_tokens > 1
            and self.latent_tokens < customers.size(1)
            and customers.size(1) >= self.latent_min_nodes
        )
        if use_bottleneck:
            self.cust_enc = self._encode_customers_bottleneck(cust_emb, customers, mask)
        else:
            self.cust_enc = self.cust_encoder(cust_emb, mask, coords = customers[:, :, :2]) #.size() = N x L_c x D
        self.cust_repr = self.dropout(self.cust_project(self.cust_enc)) #.size() = N x L_c x D
        if mask is not None:
            self.cust_repr[mask] = 0
        self._forward_profiler.add("encode_customers", time.perf_counter() - start)


    def _encode_fleet(self, vehicles, cust_repr, mask = None):
        fleet_enc = self.fleet_encoder(vehicles, cust_repr, mask = mask)
        return self.veh_project(fleet_enc)


    def _build_fleet_edges(self, vehicles):
        start = time.perf_counter()
        pos = vehicles[:, :, :2]
        time_feat = vehicles[:, :, 3:4]
        capa = vehicles[:, :, 2:3]
        dist = torch.cdist(pos, pos)
        time_gap = (time_feat - time_feat.transpose(1, 2)).abs()
        capa_gap = (capa - capa.transpose(1, 2)).abs()
        edge = torch.stack((dist, time_gap.squeeze(-1), capa_gap.squeeze(-1)), dim = -1)
        self._forward_profiler.add("build_fleet_edges", time.perf_counter() - start)
        return edge


    def _build_edge_features(self, vehicles, customers, veh_idx, veh_mask = None):
        start = time.perf_counter()
        v = vehicles.gather(1, veh_idx[:, :, None].expand(-1, -1, vehicles.size(-1)))
        v_pos = v[:, :, :2]
        v_time = v[:, :, 3:4]
        v_capa = v[:, :, 2:3]

        c_pos = customers[:, :, :2]
        dist = torch.cdist(v_pos, c_pos)
        speed = getattr(self, "veh_speed", None)
        if speed is None:
            speed = 1.0
        travel_time = dist / speed
        arrival = v_time.expand_as(travel_time) + travel_time

        if customers.size(-1) >= 5:
            tw_start = customers[:, :, 3:4].transpose(1, 2)
            tw_end = customers[:, :, 4:5].transpose(1, 2)
            wait = (tw_start - arrival).clamp(min = 0).unsqueeze(-1)
            late = (arrival - tw_end).clamp(min = 0).unsqueeze(-1)
            slack = (tw_end - arrival).unsqueeze(-1)
        else:
            wait = dist.new_zeros(dist.size()).unsqueeze(-1)
            late = dist.new_zeros(dist.size()).unsqueeze(-1)
            slack = dist.new_zeros(dist.size()).unsqueeze(-1)

        cust_demand = customers[:, :, 2:3].transpose(1, 2)
        cap_gap = v_capa.expand_as(cust_demand) - cust_demand
        if veh_mask is None:
            feasible = dist.new_ones(dist.size()).unsqueeze(-1)
        else:
            feasible = (~veh_mask).float().unsqueeze(-1)

        feats = torch.cat((
            dist.unsqueeze(-1),
            travel_time.unsqueeze(-1),
            arrival.unsqueeze(-1),
            wait,
            late,
            slack,
            feasible,
            cap_gap.unsqueeze(-1),
        ), dim = -1)
        self._forward_profiler.add("build_edge_features", time.perf_counter() - start)
        return feats


    def _repr_vehicle(self, vehicles, veh_idx, mask):
        r"""
        :param vehicles: :math:`N \times L_v \times D_v` tensor containing minibatch of vehicles' states
        :param veh_idx:  :math:`N \times 1` tensor containing minibatch of indices corresponding to currently acting vehicle
        :param mask:     :math:`N \times 1 \times L_c` tensor containing minibatch of masks
                where :math:`m_{nij} = 1` if vehicle :math:`i` cannot serve customer :math:`j` in sample :math:`n`, 0 otherwise

        :return:         :math:`N \times 1 \times D` tensor containing minibatch of representations for currently acting vehicle
        """
        start = time.perf_counter()
        veh_state = vehicles.gather(1, veh_idx[:, :, None].expand(-1, -1, vehicles.size(-1))) #.size() = N x 1 x D_v
        if mask is not None and mask.dim() == 3 and mask.size(1) != 1:
            veh_mask = mask.gather(1, veh_idx[:, :, None].expand(-1, -1, mask.size(-1)))
        else:
            veh_mask = mask
        veh_query = self._encode_fleet(veh_state, self.cust_repr, mask = veh_mask) #.size() = N x 1 x D
        self._forward_profiler.add("represent_vehicle", time.perf_counter() - start)
        return veh_query


    def _score_customers(self, veh_repr, cust_repr, edge_emb, owner_bias, lookahead, veh_mask = None):
        r"""
        :param veh_repr: :math:`N \times 1 \times D` tensor containing minibatch of representations for currently acting vehicle

        :return:         :math:`N \times 1 \times L_c` tensor containing minibatch of compatibility scores between currently acting vehicle and each customer
        """
        start = time.perf_counter()
        att_score = self.cross_fusion(veh_repr, cust_repr, edge_emb)
        
        # Normalize each score source to a common scale (mean=0, std=1) across candidates
        # so they contribute "fairly" to the final decision.
        def z_norm(s):
            return (s - s.mean(dim = -1, keepdim = True)) / (s.std(dim = -1, keepdim = True) + 1e-8)

        # Concatenate normalized score sources
        # Shape: (N, 1, L_c, 3)
        att_norm = z_norm(att_score)
        owner_norm = z_norm(owner_bias)
        look_norm = z_norm(lookahead)

        combined_scores = torch.stack([
            att_norm,
            owner_norm,
            look_norm
        ], dim = -1)
        
        # Non-linear fusion using the score_fusion MLP
        compat_base = self.score_fusion(combined_scores).squeeze(-1)

        if self.sbg_moe_enable:
            moe_start = time.perf_counter()
            # Learned experts: replace fixed linear combinations of [att, owner, look]
            expert_local  = self.expert_local_net(combined_scores).squeeze(-1)   # (N, 1, L_c)
            expert_future = self.expert_future_net(combined_scores).squeeze(-1)  # (N, 1, L_c)

            if veh_mask is not None:
                feasible_ratio = (~veh_mask).float().mean(dim = -1, keepdim = True)
            else:
                feasible_ratio = compat_base.new_full((compat_base.size(0), 1, 1), 0.5)
            look_std = look_norm.std(dim = -1, keepdim = True)

            if self.sbg_moe_uncertainty:
                if veh_mask is not None:
                    masked_base = compat_base.masked_fill(veh_mask, -float('inf'))
                    all_masked = veh_mask.all(dim = -1, keepdim = True)
                    if all_masked.any():
                        masked_base = masked_base.clone()
                        fallback_rows = all_masked.squeeze(-1).squeeze(-1)
                        masked_base[fallback_rows, :, 0] = 0.0
                else:
                    masked_base = compat_base

                probs = masked_base.softmax(dim = -1)
                entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim = -1, keepdim = True)
                norm_entropy = entropy / max(1.0, torch.log(torch.tensor(float(compat_base.size(-1)), device = compat_base.device)).item())

                top2 = masked_base.topk(k = min(2, masked_base.size(-1)), dim = -1).values
                if top2.size(-1) == 2:
                    margin = (top2[..., 0:1] - top2[..., 1:2]).clamp(min = 0.0)
                else:
                    margin = masked_base.new_zeros(masked_base.size(0), 1, 1)

                confident = (norm_entropy < self.sbg_moe_entropy_floor) | (margin > self.sbg_moe_margin_ceil)
                moe_active = (~confident).float()
                margin_scale = (1.0 / (1.0 + margin)).clamp(min = 0.0, max = 1.0)
                dyn = norm_entropy.clamp(min = 0.0, max = 1.0) * margin_scale
                dyn_strength = self.sbg_moe_min_strength + (self.sbg_moe_strength - self.sbg_moe_min_strength) * dyn
                dyn_strength = dyn_strength * moe_active
            else:
                dyn_strength = compat_base.new_full((compat_base.size(0), 1, 1), self.sbg_moe_strength)

            # Learned router: replaces rule-based (feasible_ratio<0.35)/(look_std>0.9) flags
            router_context = torch.cat([feasible_ratio, look_std], dim = -1)   # (N, 1, 2)
            router_logits  = self.moe_router(router_context)                    # (N, 1, 2)
            router_probs   = router_logits.softmax(dim = -1)                    # (N, 1, 2)
            alpha_local  = router_probs[..., 0:1] * dyn_strength               # (N, 1, 1)
            alpha_future = router_probs[..., 1:2] * dyn_strength               # (N, 1, 1)

            # Load balancing auxiliary loss: maximise routing entropy to prevent expert collapse
            router_entropy    = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim = -1)
            load_balance_loss = -router_entropy.mean()
            if self._moe_aux_loss is None:
                self._moe_aux_loss = load_balance_loss
            else:
                self._moe_aux_loss = self._moe_aux_loss + load_balance_loss

            delta_local = (expert_local - compat_base).clamp(min = -2.0, max = 2.0)
            delta_future = (expert_future - compat_base).clamp(min = -2.0, max = 2.0)

            compat = compat_base + alpha_local * delta_local + alpha_future * delta_future
            self._forward_profiler.add("sbg_moe", time.perf_counter() - moe_start)
        else:
            compat = compat_base

        if self.tanh_xplor is not None:
            compat = self.tanh_xplor * compat.tanh()
        self._forward_profiler.add("score_customers", time.perf_counter() - start)
        return compat


    def _resolve_sbg_k(self, cand_count, feasible_count):
        if not self.sbg_enable or self.sbg_cand_k is None or self.sbg_cand_k <= 0:
            return cand_count

        k = min(self.sbg_cand_k, cand_count)
        if self.sbg_adaptive_k:
            mean_feasible = int(feasible_count.float().mean().item())
            ratio = mean_feasible / max(1, cand_count)
            if ratio > 0.6:
                k = int(round(k * 1.5))
            elif ratio < 0.3:
                k = int(round(k * 0.75))
        k = max(1, k)
        k = max(self.sbg_k_min, k)
        k_max = cand_count if self.sbg_k_max is None else self.sbg_k_max
        k = min(k, k_max, cand_count)
        return k


    def _select_sbg_candidates(self, edge_feat, owner_bias, veh_mask):
        start = time.perf_counter()
        full_count = edge_feat.size(2)
        veh_mask_2d = veh_mask.squeeze(1)
        feasible_count = (~veh_mask_2d).sum(dim = 1)
        k = self._resolve_sbg_k(full_count, feasible_count)
        if k >= full_count:
            full_idx = torch.arange(full_count, device = edge_feat.device).unsqueeze(0).expand(edge_feat.size(0), -1)
            self._forward_profiler.add("sbg_select", time.perf_counter() - start)
            return full_idx, veh_mask

        dist = edge_feat[..., 0].squeeze(-1).squeeze(1)
        late = edge_feat[..., 4].squeeze(-1).squeeze(1)
        slack = edge_feat[..., 5].squeeze(-1).squeeze(1)
        owner = owner_bias.squeeze(1)

        cheap_score = -dist - self.sbg_late_penalty * late + self.sbg_slack_weight * slack + self.sbg_owner_weight * owner
        cheap_score = cheap_score.masked_fill(veh_mask_2d, -float('inf'))

        all_masked = veh_mask_2d.all(dim = 1)
        if all_masked.any():
            cheap_score = cheap_score.clone()
            cheap_score[all_masked, 0] = 0.0

        cand_idx = cheap_score.topk(k, dim = 1, largest = True).indices
        cand_mask = veh_mask.gather(2, cand_idx[:, None, :])
        if all_masked.any():
            cand_mask = cand_mask.clone()
            cand_mask[all_masked, :, 0] = False
        self._forward_profiler.add("sbg_select", time.perf_counter() - start)
        return cand_idx, cand_mask


    def _get_logp(self, compat : torch.Tensor, veh_mask):
        r"""
        :param compat:   :math:`N \times 1 \times L_c` tensor containing minibatch of compatibility scores between currently acting vehicle and each customer
        :param veh_mask: :math:`N \times 1 \times L_c` tensor containing minibatch of masks
                where :math:`m_{nj} = 1` if currently acting vehicle cannot serve customer :math:`j` in sample :math:`n`, 0 otherwise

        :return:         :math:`N \times L_c` tensor containing minibatch of log-probabilities for choosing which customer to serve next
        """
        start = time.perf_counter()
        mask = veh_mask
        all_masked = mask.all(dim = 2, keepdim = True)
        if all_masked.any():
            mask = mask.clone()
            all_masked_rows = all_masked.squeeze(2).squeeze(1)
            mask[all_masked_rows, 0, 0] = False
        compat = compat.clone()
        compat[mask] = -float('inf')
        logprobs = compat.log_softmax(dim = 2).squeeze(1)
        self._forward_profiler.add("logp", time.perf_counter() - start)
        return logprobs


    def step(self, dyna):
        
        veh_repr = self._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.cur_veh_mask)
        
        edge_feat = self._build_edge_features(dyna.vehicles, dyna.nodes, dyna.cur_veh_idx, dyna.cur_veh_mask)
        edge_emb = self.edge_encoder(edge_feat)

        owner_start = time.perf_counter()
        owner_logits = self.owner_head(self._veh_memory, self.cust_repr)
        owner_prob = owner_logits.softmax(dim = 1)
        owner_bias = owner_prob.gather(1, dyna.cur_veh_idx[:, :, None].expand(-1, -1, owner_prob.size(-1)))
        owner_bias = owner_bias.clamp_min(1e-9).log()
        self._forward_profiler.add("ownership_head", time.perf_counter() - owner_start)

        use_sbg = self.sbg_enable and self.sbg_cand_k is not None and self.sbg_cand_k > 0
        if use_sbg:
            cand_idx, cand_mask = self._select_sbg_candidates(edge_feat, owner_bias, dyna.cur_veh_mask)
            cust_repr_scored = self.cust_repr.gather(1, cand_idx[:, :, None].expand(-1, -1, self.cust_repr.size(-1)))
            edge_emb_scored = edge_emb.gather(2, cand_idx[:, None, :, None].expand(-1, -1, -1, edge_emb.size(-1)))
            owner_bias_scored = owner_bias.gather(2, cand_idx[:, None, :])
        else:
            cand_idx = None
            cand_mask = dyna.cur_veh_mask
            cust_repr_scored = self.cust_repr
            edge_emb_scored = edge_emb
            owner_bias_scored = owner_bias

        lookahead_start = time.perf_counter()
        lookahead = self.lookahead_head(veh_repr, cust_repr_scored, edge_emb_scored)
        self._forward_profiler.add("lookahead_head", time.perf_counter() - lookahead_start)
        compat : torch.Tensor = self._score_customers(
            veh_repr,
            cust_repr_scored,
            edge_emb_scored,
            owner_bias_scored,
            lookahead,
            cand_mask,
        )
        logp_local = self._get_logp(compat, cand_mask)

        if cand_idx is None:
            logp = logp_local
        else:
            logp = logp_local.new_full((logp_local.size(0), dyna.cur_veh_mask.size(-1)), -float('inf'))
            logp.scatter_(1, cand_idx, logp_local)
        if self.greedy:
            cust_idx = logp.argmax(dim = 1, keepdim = True)
        else:
            cust_idx = logp.exp().multinomial(1)
        self._update_memory(dyna.cur_veh_idx, cust_idx, veh_repr, edge_emb)
        return cust_idx, logp.gather(1, cust_idx)


    def _update_memory(self, veh_idx, cust_idx, veh_repr, edge_emb):
        start = time.perf_counter()
        edge_sel = edge_emb.gather(2, cust_idx[:, :, None, None].expand(-1, -1, -1, edge_emb.size(-1)))
        cust_sel = self.cust_repr.gather(1, cust_idx[:, :, None].expand(-1, -1, self.cust_repr.size(-1)))
        self._veh_memory = self.coord_memory.update(self._veh_memory, veh_idx, veh_repr, cust_sel, edge_sel)
        self._forward_profiler.add("memory_update", time.perf_counter() - start)


    def forward(self, dyna):
        self._forward_profiler.reset()
        self._moe_aux_loss = None
        dyna.reset()
        actions, logps, rewards = [], [], []
        if hasattr(dyna, "veh_speed"):
            self.veh_speed = dyna.veh_speed
        self._reset_memory(dyna)
        while not dyna.done:
            if dyna.new_customers:
                self._encode_customers(dyna.nodes, dyna.cust_mask)
            cust_idx, logp = self.step(dyna)
            actions.append( (dyna.cur_veh_idx, cust_idx) )
            logps.append( logp )
            rewards.append( dyna.step(cust_idx) )
        return actions, logps, rewards


    def _reset_memory(self, dyna):
        self._veh_memory = dyna.vehicles.new_zeros((dyna.vehicles.size(0), dyna.vehicles.size(1), self.coord_memory.hidden_size))

    def reset_forward_profiling(self):
        self._forward_profiler.reset()

    def get_forward_profiling_summary(self):
        return self._forward_profiler.get_summary()
