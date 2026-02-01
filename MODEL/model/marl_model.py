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

class EdgeEnhencedLearner(nn.Module):
    def __init__(self, cust_feat_size, veh_state_size, model_size = 128,
            layer_count = 3, head_count = 8, ff_size = 512, tanh_xplor = 10, greedy = False,
            edge_feat_size = 8, cust_k = 10, memory_size = None, lookahead_hidden = 128,
            dropout = 0.1
            ):
        r"""
        :param model_size:  Dimension :math:`D` shared by all intermediate layers
        :param layer_count: Number of layers in customers' (graph) Transformer Encoder
        :param head_count:  Number of heads in all Multi-Head Attention layers
        :param ff_size:     Dimension of the feed-forward sublayers in Transformer Encoder
        :param tanh_xplor:  Enable tanh exploration and set its amplitude
        """
        super().__init__()

        self.model_size = model_size

        self.tanh_xplor = tanh_xplor

        self.depot_embedding = nn.Linear(cust_feat_size, model_size)
        self.cust_embedding  = nn.Linear(cust_feat_size, model_size)
        self.cust_encoder    = GraphEncoder(layer_count, head_count, model_size, ff_size, k = cust_k, dropout = dropout)

        self.fleet_encoder   = FleetEncoder(layer_count, head_count, model_size, ff_size, dropout = dropout)
        self.edge_encoder    = EdgeFeatureEncoder(edge_feat_size, model_size, dropout = dropout)
        self.cross_fusion    = CrossEdgeFusion(head_count, model_size)

        self.coord_memory    = CoordinationMemory(veh_state_size, model_size if memory_size is None else memory_size, dropout = dropout)
        self.owner_head      = OwnershipHead(model_size, dropout = dropout)
        self.lookahead_head  = LookaheadHead(model_size, hidden_size = lookahead_hidden, dropout = dropout)

        self.cust_project    = nn.Linear(model_size, model_size)
        self.veh_project     = nn.Linear(model_size, model_size)
        self.fusion_norm     = nn.LayerNorm(3)
        self.score_fusion    = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.dropout         = nn.Dropout(dropout)
        self.greedy = greedy


    def _encode_customers(self, customers, mask = None):
        r"""
        :param customers: :math:`N \times L_c \times D_c` tensor containing minibatch of customers' features
        :param mask:      :math:`N \times L_c` tensor containing minibatch of masks
                where :math:`m_{nj} = 1` if customer :math:`j` in sample :math:`n` is hidden (pad or dyn), 0 otherwise
        """
        cust_emb = torch.cat((
            self.depot_embedding(customers[:,0:1,:]),
            self.cust_embedding(customers[:,1:,:])
            ), dim = 1) #.size() = N x L_c x D
        if mask is not None:
            cust_emb = cust_emb.masked_fill(mask.unsqueeze(-1), 0.0)
        cust_emb = self.dropout(cust_emb)
        self.cust_enc = self.cust_encoder(cust_emb, mask, coords = customers[:, :, :2]) #.size() = N x L_c x D
        self.cust_repr = self.dropout(self.cust_project(self.cust_enc)) #.size() = N x L_c x D
        if mask is not None:
            self.cust_repr = self.cust_repr.masked_fill(mask.unsqueeze(-1), 0.0)


    def _encode_fleet(self, vehicles, cust_repr, mask = None):
        fleet_enc = self.fleet_encoder(vehicles, cust_repr, mask = mask)
        return self.dropout(self.veh_project(fleet_enc))


    def _build_fleet_edges(self, vehicles):
        pos = vehicles[:, :, :2]
        time = vehicles[:, :, 3:4]
        capa = vehicles[:, :, 2:3]
        dist = torch.cdist(pos, pos)
        time_gap = (time - time.transpose(1, 2)).abs()
        capa_gap = (capa - capa.transpose(1, 2)).abs()
        return torch.stack((dist, time_gap.squeeze(-1), capa_gap.squeeze(-1)), dim = -1)


    def _build_edge_features(self, vehicles, customers, veh_idx, veh_mask = None):
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
        return feats


    def _repr_vehicle(self, vehicles, veh_idx, mask):
        r"""
        :param vehicles: :math:`N \times L_v \times D_v` tensor containing minibatch of vehicles' states
        :param veh_idx:  :math:`N \times 1` tensor containing minibatch of indices corresponding to currently acting vehicle
        :param mask:     :math:`N \times L_v \times L_c` tensor containing minibatch of masks
                where :math:`m_{nij} = 1` if vehicle :math:`i` cannot serve customer :math:`j` in sample :math:`n`, 0 otherwise

        :return:         :math:`N \times 1 \times D` tensor containing minibatch of representations for currently acting vehicle
        """
        fleet_repr = self._encode_fleet(vehicles, self.cust_repr, mask = mask) #.size() = N x L_v x D
        veh_query = fleet_repr.gather(1, veh_idx.unsqueeze(2).expand(-1, -1, self.model_size)) #.size() = N x 1 x D
        return veh_query


    def _score_customers(self, veh_repr, cust_repr, edge_emb, owner_bias, lookahead):
        r"""
        :param veh_repr: :math:`N \times 1 \times D` tensor containing minibatch of representations for currently acting vehicle

        :return:         :math:`N \times 1 \times L_c` tensor containing minibatch of compatibility scores between currently acting vehicle and each customer
        """
        att_score = self.dropout(self.cross_fusion(veh_repr, cust_repr, edge_emb))
        
        # Non-linear fusion of scores
        scores = torch.cat((
            att_score.unsqueeze(-1),
            owner_bias.unsqueeze(-1),
            lookahead.unsqueeze(-1)
        ), dim = -1) # N x 1 x L_c x 3
        
        # Use LayerNorm to stabilize scores before MLP
        scores = self.fusion_norm(scores)
        compat = self.score_fusion(scores).squeeze(-1) # N x 1 x L_c
        
        if self.tanh_xplor is not None:
            compat = self.tanh_xplor * compat.tanh()
        return compat


    def _get_logp(self, compat : torch.Tensor, veh_mask):
        r"""
        :param compat:   :math:`N \times 1 \times L_c` tensor containing minibatch of compatibility scores between currently acting vehicle and each customer
        :param veh_mask: :math:`N \times 1 \times L_c` tensor containing minibatch of masks
                where :math:`m_{nj} = 1` if currently acting vehicle cannot serve customer :math:`j` in sample :math:`n`, 0 otherwise

        :return:         :math:`N \times L_c` tensor containing minibatch of log-probabilities for choosing which customer to serve next
        """
        if veh_mask is not None:
            # Handle all-masked rows to prevent nan in log_softmax
            all_masked = veh_mask.all(dim = 2, keepdim = True)
            if all_masked.any():
                # If all nodes are masked, temporarily unmask depot (index 0) 
                # to prevent nan, although environment should ideally prevent this.
                veh_mask = veh_mask.clone()
                veh_mask[:, :, 0] = veh_mask[:, :, 0] & (~all_masked.squeeze(-1))
            
            compat = compat.masked_fill(veh_mask, float('-inf'))
            
        return compat.log_softmax(dim = 2).squeeze(1)


    def step(self, dyna):
        
        veh_repr = self._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.mask)
        
        edge_feat = self._build_edge_features(dyna.vehicles, dyna.nodes, dyna.cur_veh_idx, dyna.cur_veh_mask)
        edge_emb = self.edge_encoder(edge_feat)

        owner_logits = self.owner_head(self._veh_memory, self.cust_repr)
        owner_prob = owner_logits.softmax(dim = 1)
        owner_bias = owner_prob.gather(1, dyna.cur_veh_idx[:, :, None].expand(-1, -1, owner_prob.size(-1)))
        # Use a safer epsilon for log to prevent gradient explosion in float16/AMP
        owner_bias = torch.log(owner_bias + 1e-5)

        lookahead = self.lookahead_head(veh_repr, self.cust_repr, edge_emb)
        compat : torch.Tensor = self._score_customers(veh_repr, self.cust_repr, edge_emb, owner_bias, lookahead)
        logp = self._get_logp(compat, dyna.cur_veh_mask)
        if self.greedy:
            cust_idx = logp.argmax(dim = 1, keepdim = True)
        else:
            cust_idx = logp.exp().multinomial(1)
        self._update_memory(dyna.cur_veh_idx, cust_idx, veh_repr, edge_emb)
        return cust_idx, logp.gather(1, cust_idx)


    def _update_memory(self, veh_idx, cust_idx, veh_repr, edge_emb):
        edge_sel = edge_emb.gather(2, cust_idx[:, :, None, None].expand(-1, -1, -1, edge_emb.size(-1)))
        cust_sel = self.cust_repr.gather(1, cust_idx[:, :, None].expand(-1, -1, self.cust_repr.size(-1)))
        self._veh_memory = self.coord_memory.update(self._veh_memory, veh_idx, veh_repr, cust_sel, edge_sel)


    def forward(self, dyna):
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
