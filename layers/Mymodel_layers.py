import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ._mha import _MHA_V2


# ============================================================================
# VECTRA Graph Encoder: Feature-Disentangled Anisotropic Graph Transformer
# ============================================================================
#
# Theoretical Motivation
# ----------------------
# In DVRPTW the complete customer graph carries a rich multi-relational
# structure.  Two customers are "routing-compatible" only when they satisfy
# MULTIPLE criteria simultaneously:
#
#   1. Spatial proximity         – minimizes travel cost / time
#   2. Temporal compatibility    – time windows must permit sequential visits
#   3. Demand coupling           – combined demands must fit vehicle capacity
#
# Existing graph encoders for VRP use only Euclidean distance as edge
# information, collapsing all relational signals into a single scalar.
# VECTRA explicitly factorises these relations and feeds them into specialised
# attention heads, yielding a strictly more expressive encoder:
#
# (a) Multi-Relational Pairwise Features
#     Compute spatial, temporal and demand pairwise features between every
#     pair of nodes, producing an edge-feature tensor  E ∈ R^{L×L×R}.
#
# (b) Disentangled Attention Heads
#     Partition heads into *groups*, each receiving bias from a different
#     relation type.  This is equivalent to learning a soft mixture of
#     relation-specific graph kernels while preserving full expressiveness
#     through "cross-relation" heads that observe all features jointly.
#
# (c) Edge-Valued Message Passing
#     Standard edge-biased attention modulates only WHO is attended to.
#     VECTRA additionally injects edge features into the VALUE stream,
#     controlling WHAT information flows along each edge:
#         output_i = Σ_j  α_ij · (W_v · h_j  +  g · W_e · e_ij)
#     This is strictly more expressive than additive attention bias alone.
#
# (d) Node Structural Encoding
#     Per-node features (centrality, TW urgency, demand significance)
#     provide *positional* context analogous to graph positional encodings
#     but grounded in DVRPTW semantics.
# ============================================================================


class VECTRAPairwiseFeatures(nn.Module):
	"""Compute multi-relational pairwise features from raw DVRPTW node attributes.

	Produces ``(N, L, L, 6)`` features capturing three relationship types:

	* **Spatial** (index 0):  normalised Euclidean distance
	* **Temporal** (indices 1-3):  TW overlap ratio, forward precedence,
	  backward precedence
	* **Demand** (indices 4-5):  normalised demand difference, combined
	  load ratio

	All features are normalised to roughly [0, 1] for stable training.
	"""

	SPATIAL_SLICE  = slice(0, 1)   # [0]
	TEMPORAL_SLICE = slice(1, 4)   # [1, 2, 3]
	DEMAND_SLICE   = slice(4, 6)   # [4, 5]
	TOTAL_DIM      = 6

	def __init__(self, temporal_temperature = 0.1):
		super().__init__()
		self.temporal_temperature = temporal_temperature

	def forward(self, raw_nodes):
		"""
		:param raw_nodes: ``(N, L, F)`` with F >= 3; columns ``[x, y, demand, tw_start, tw_end, ...]``
		:return:          ``(N, L, L, 6)`` pairwise features
		"""
		coords = raw_nodes[:, :, :2]
		demand = raw_nodes[:, :, 2:3]

		# ---- Spatial: normalised distance ----
		dist = torch.cdist(coords, coords)
		max_dist = dist.amax(dim = (-1, -2), keepdim = True).clamp(min = 1e-6)
		dist_norm = dist / max_dist                              # (N, L, L)

		# ---- Temporal features ----
		has_tw = raw_nodes.size(-1) >= 5
		if has_tw:
			tw_s = raw_nodes[:, :, 3:4]                          # (N, L, 1)
			tw_e = raw_nodes[:, :, 4:5]

			# normalise TW to [0, ~1] so temperature is universal
			tw_min   = tw_s.amin(dim = 1, keepdim = True)
			tw_range = (tw_e.amax(dim = 1, keepdim = True) - tw_min).clamp(min = 1e-6)
			tw_s_n   = (tw_s - tw_min) / tw_range
			tw_e_n   = (tw_e - tw_min) / tw_range

			tw_s_i, tw_e_i = tw_s_n.unsqueeze(2), tw_e_n.unsqueeze(2)  # (N,L,1,1)
			tw_s_j, tw_e_j = tw_s_n.unsqueeze(1), tw_e_n.unsqueeze(1)  # (N,1,L,1)

			# overlap ratio (symmetric)
			overlap   = (torch.min(tw_e_i, tw_e_j) - torch.max(tw_s_i, tw_s_j)).clamp(min = 0)
			tw_width  = (tw_e_n - tw_s_n).clamp(min = 1e-6)
			max_width = torch.max(tw_width.unsqueeze(2), tw_width.unsqueeze(1))
			overlap_ratio = (overlap / max_width).squeeze(-1)           # (N, L, L)

			# forward precedence: P(j can follow i)
			fwd_prec = torch.sigmoid(
				(tw_s_j - tw_e_i).squeeze(-1) / self.temporal_temperature)

			# backward precedence: P(i can follow j)
			bwd_prec = torch.sigmoid(
				(tw_s_i - tw_e_j).squeeze(-1) / self.temporal_temperature)
		else:
			N, L = raw_nodes.size(0), raw_nodes.size(1)
			overlap_ratio = dist_norm.new_zeros(N, L, L)
			fwd_prec      = dist_norm.new_zeros(N, L, L)
			bwd_prec      = dist_norm.new_zeros(N, L, L)

		# ---- Demand features ----
		max_demand = demand.amax(dim = 1, keepdim = True).clamp(min = 1e-6)
		demand_n   = demand / max_demand
		demand_diff    = (demand_n.unsqueeze(2) - demand_n.unsqueeze(1)).abs().squeeze(-1)
		combined_load  = ((demand_n.unsqueeze(2) + demand_n.unsqueeze(1)) / 2).squeeze(-1)

		return torch.stack([
			dist_norm,       # [0] spatial
			overlap_ratio,   # [1] temporal
			fwd_prec,        # [2] temporal
			bwd_prec,        # [3] temporal
			demand_diff,     # [4] demand
			combined_load,   # [5] demand
		], dim = -1)


class NodeStructuralEncoding(nn.Module):
	"""Problem-aware per-node structural features added to initial embeddings.

	Computes four features that capture a node's *role* in the graph:

	* **Centrality** – mean distance to all other nodes (spatial)
	* **TW urgency** – inverse TW width, normalised (scheduling pressure)
	* **TW midpoint** – centre of time window, normalised (temporal position)
	* **Demand significance** – normalised demand value
	"""

	def __init__(self, model_size):
		super().__init__()
		self.proj = nn.Sequential(
			nn.Linear(4, model_size),
			nn.GELU(),
			nn.Linear(model_size, model_size),
		)

	def forward(self, raw_nodes):
		"""
		:param raw_nodes: ``(N, L, F)`` with F >= 3
		:return:          ``(N, L, D)``
		"""
		coords = raw_nodes[:, :, :2]
		demand = raw_nodes[:, :, 2:3]

		# centrality
		dist = torch.cdist(coords, coords)
		centrality = dist.mean(dim = -1, keepdim = True)
		centrality = centrality / centrality.amax(dim = 1, keepdim = True).clamp(min = 1e-6)

		# demand significance
		demand_sig = demand / demand.amax(dim = 1, keepdim = True).clamp(min = 1e-6)

		has_tw = raw_nodes.size(-1) >= 5
		if has_tw:
			tw_s = raw_nodes[:, :, 3:4]
			tw_e = raw_nodes[:, :, 4:5]
			tw_width = (tw_e - tw_s).clamp(min = 1e-6)
			urgency  = 1.0 / tw_width
			urgency  = urgency / urgency.amax(dim = 1, keepdim = True).clamp(min = 1e-6)
			midpoint = (tw_s + tw_e) / 2
			tw_min   = tw_s.amin(dim = 1, keepdim = True)
			tw_range = (tw_e.amax(dim = 1, keepdim = True) - tw_min).clamp(min = 1e-6)
			midpoint = (midpoint - tw_min) / tw_range
		else:
			urgency  = demand.new_zeros(demand.size())
			midpoint = demand.new_zeros(demand.size())

		feats = torch.cat([centrality, demand_sig, urgency, midpoint], dim = -1)
		return self.proj(feats)


class DisentangledRelationBias(nn.Module):
	"""Project relation-specific edge features to per-head attention biases.

	Heads are partitioned into four groups::

	    ┌──────────┬──────────────┬────────────────────┐
	    │  Group   │  Features    │  Captures           │
	    ├──────────┼──────────────┼────────────────────┤
	    │ Spatial  │ distance     │ proximity patterns  │
	    │ Temporal │ overlap/prec │ TW compatibility    │
	    │ Demand   │ diff/load    │ capacity coupling   │
	    │ Cross    │ ALL          │ arbitrary combos    │
	    └──────────┴──────────────┴────────────────────┘

	Each group's MLP maps its feature subset to scalar biases for its heads.
	"""

	def __init__(self, head_count, spatial_heads = None, temporal_heads = None,
				 demand_heads = None, hidden_dim = 32):
		super().__init__()
		self.head_count = head_count

		# auto-allocate head budget
		if spatial_heads is None:
			base      = head_count // 4
			remainder = head_count % 4
			spatial_heads  = base + (1 if remainder > 0 else 0)
			temporal_heads = base + (1 if remainder > 1 else 0)
			demand_heads   = base + (1 if remainder > 2 else 0)
		if temporal_heads is None:
			temporal_heads = (head_count - spatial_heads) // 3
		if demand_heads is None:
			demand_heads = (head_count - spatial_heads - temporal_heads) // 2

		cross_heads = head_count - spatial_heads - temporal_heads - demand_heads
		assert cross_heads >= 0, (
			f"Head budget exceeded: {spatial_heads}+{temporal_heads}+"
			f"{demand_heads} > {head_count}")

		self.group_sizes = [spatial_heads, temporal_heads, demand_heads, cross_heads]

		# relation-specific MLPs
		self.spatial_mlp = nn.Sequential(
			nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, spatial_heads)
		) if spatial_heads > 0 else None
		self.temporal_mlp = nn.Sequential(
			nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, temporal_heads)
		) if temporal_heads > 0 else None
		self.demand_mlp = nn.Sequential(
			nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, demand_heads)
		) if demand_heads > 0 else None
		self.cross_mlp = nn.Sequential(
			nn.Linear(VECTRAPairwiseFeatures.TOTAL_DIM, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, cross_heads)
		) if cross_heads > 0 else None

	def forward(self, edge_features):
		"""
		:param edge_features: ``(N, L, L, 6)``
		:return:              ``(N, H, L, L)`` per-head bias
		"""
		parts = []
		if self.spatial_mlp  is not None:
			parts.append(self.spatial_mlp(edge_features[..., VECTRAPairwiseFeatures.SPATIAL_SLICE]))
		if self.temporal_mlp is not None:
			parts.append(self.temporal_mlp(edge_features[..., VECTRAPairwiseFeatures.TEMPORAL_SLICE]))
		if self.demand_mlp   is not None:
			parts.append(self.demand_mlp(edge_features[..., VECTRAPairwiseFeatures.DEMAND_SLICE]))
		if self.cross_mlp    is not None:
			parts.append(self.cross_mlp(edge_features))
		return torch.cat(parts, dim = -1).permute(0, 3, 1, 2)


class VECTRAGraphEncoderLayer(nn.Module):
	"""Single layer of the VECTRA Graph Transformer.

	Combines three mechanisms:

	1. **Disentangled multi-relational edge bias** – each head group receives
	   bias from a *different* relation type (spatial / temporal / demand /
	   cross), enabling specialisation while maintaining full expressiveness.

	2. **Edge-valued message passing** – beyond controlling *who* is attended
	   to, edge features also modulate *what* information flows between nodes::

	       output_i  =  Σ_j  α_ij · ( W_v·h_j  +  σ(g) · W_e·e_ij )

	   where ``g`` is a learnable per-head gate (initialised near-zero).

	3. **Post-LayerNorm** with GELU FFN for consistency with the existing
	   model backbone.
	"""

	def __init__(self, head_count, model_size, ff_size,
				 edge_feat_dim = 6, spatial_heads = None, temporal_heads = None,
				 demand_heads = None, bias_hidden = 32, dropout = 0.1,
				 use_edge_values = True):
		super().__init__()
		self.head_count  = head_count
		self.model_size  = model_size
		self.head_dim    = model_size // head_count
		self.use_edge_values = use_edge_values
		assert model_size % head_count == 0

		# QKV
		self.q_proj   = nn.Linear(model_size, model_size, bias = False)
		self.k_proj   = nn.Linear(model_size, model_size, bias = False)
		self.v_proj   = nn.Linear(model_size, model_size, bias = False)
		self.out_proj = nn.Linear(model_size, model_size, bias = False)

		# disentangled edge bias
		self.edge_bias = DisentangledRelationBias(
			head_count, spatial_heads, temporal_heads, demand_heads, bias_hidden)

		# edge-valued message passing  (gate starts mostly closed ≈ 0.12)
		if use_edge_values:
			self.edge_value_proj = nn.Sequential(
				nn.Linear(edge_feat_dim, model_size // 2),
				nn.GELU(),
				nn.Linear(model_size // 2, model_size),
			)
			self.edge_gate = nn.Parameter(torch.full((1, head_count, 1, 1), -2.0))

		# norms + FFN
		self.norm1      = nn.LayerNorm(model_size)
		self.ff1        = nn.Linear(model_size, ff_size)
		self.ff2        = nn.Linear(ff_size, model_size)
		self.norm2      = nn.LayerNorm(model_size)
		self.attn_drop  = nn.Dropout(dropout)
		self.ff_drop    = nn.Dropout(dropout)

	# ------------------------------------------------------------------

	def forward(self, h_in, edge_features, mask = None):
		"""
		:param h_in:          ``(N, L, D)``
		:param edge_features: ``(N, L, L, R)``
		:param mask:          ``(N, L)`` or ``(N, L, L)`` bool – True = masked
		:return:              ``(N, L, D)``
		"""
		B, L, D = h_in.size()
		H, d    = self.head_count, self.head_dim

		q = self.q_proj(h_in).view(B, L, H, d).transpose(1, 2)   # (B,H,L,d)
		k = self.k_proj(h_in).view(B, L, H, d).transpose(1, 2)
		v = self.v_proj(h_in).view(B, L, H, d).transpose(1, 2)

		# ---- attention scores + disentangled bias ----
		scores = torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5)
		scores = scores + self.edge_bias(edge_features)

		# ---- masking ----
		if mask is not None:
			if mask.dim() == 2:
				key_mask = mask[:, None, None, :].expand(-1, H, L, -1)
			elif mask.dim() == 3:
				key_mask = mask[:, None, :, :].expand(-1, H, -1, -1)
			else:
				raise ValueError("mask must be 2-D or 3-D")
			all_masked = key_mask.all(dim = -1, keepdim = True)
			if all_masked.any():
				key_mask = key_mask.clone()
				key_mask.masked_fill_(all_masked, False)
			scores = scores.masked_fill(key_mask, float('-inf'))

		attn = F.softmax(scores, dim = -1)
		attn = self.attn_drop(attn)

		# ---- standard + edge-valued output ----
		out = torch.matmul(attn, v)                                # (B,H,L,d)

		if self.use_edge_values:
			ev = self.edge_value_proj(edge_features)               # (B,L,L,D)
			ev = ev.view(B, L, L, H, d).permute(0, 3, 1, 2, 4)   # (B,H,L,L,d)
			# memory-efficient contraction  Σ_j α_ij · e_ij
			attn_flat = attn.reshape(B * H * L, 1, L)
			ev_flat   = ev.reshape(B * H * L, L, d)
			edge_out  = torch.bmm(attn_flat, ev_flat).reshape(B, H, L, d)
			out = out + torch.sigmoid(self.edge_gate) * edge_out

		out = out.transpose(1, 2).contiguous().view(B, L, D)
		att = self.out_proj(out)

		# ---- residual + FFN  (post-LN) ----
		h = self.norm1(h_in + att)
		ff = self.ff2(F.gelu(self.ff1(h)))
		h_out = self.norm2(h + self.ff_drop(ff))

		# zero masked positions
		if mask is not None:
			if mask.dim() == 2:
				h_out[mask] = 0
			elif mask.dim() == 3:
				node_mask = mask.all(dim = -1)
				if node_mask.any():
					h_out = h_out.masked_fill(node_mask.unsqueeze(-1), 0.0)
		return h_out


class VECTRAGraphEncoder(nn.Module):
	"""Feature-Disentangled Anisotropic Graph Transformer for DVRPTW.

	End-to-end graph encoder that:

	1. Enriches initial embeddings with :class:`NodeStructuralEncoding`
	2. Extracts multi-relational pairwise features with
	   :class:`VECTRAPairwiseFeatures`
	3. Processes through stacked :class:`VECTRAGraphEncoderLayer` layers
	   with disentangled attention and edge-valued message passing

	Supports KNN sparsification (``k``) and adaptive depth.
	"""

	def __init__(self, layer_count, head_count, model_size, ff_size,
				 k = None, adaptive_depth = False, min_layers = 1, easy_ratio = 0.6,
				 spatial_heads = None, temporal_heads = None, demand_heads = None,
				 bias_hidden = 32, dropout = 0.1, temporal_temperature = 0.1,
				 use_edge_values = True):
		super().__init__()
		self.k = k
		self.adaptive_depth = adaptive_depth
		self.min_layers = max(1, min_layers)
		self.easy_ratio = easy_ratio

		self.pairwise_features   = VECTRAPairwiseFeatures(temporal_temperature)
		self.structural_encoding = NodeStructuralEncoding(model_size)

		self.layers = nn.ModuleList([
			VECTRAGraphEncoderLayer(
				head_count, model_size, ff_size,
				edge_feat_dim  = VECTRAPairwiseFeatures.TOTAL_DIM,
				spatial_heads  = spatial_heads,
				temporal_heads = temporal_heads,
				demand_heads   = demand_heads,
				bias_hidden    = bias_hidden,
				dropout        = dropout,
				use_edge_values = use_edge_values,
			) for _ in range(layer_count)
		])

	def _resolve_layer_count(self, mask, total_layers):
		if (not self.adaptive_depth) or total_layers <= 1:
			return total_layers
		min_layers = min(self.min_layers, total_layers)
		if mask is None:
			return total_layers
		visible_ratio = 1.0 - mask.float().mean().item()
		if visible_ratio >= self.easy_ratio:
			return min_layers
		span     = total_layers - min_layers
		hardness = (self.easy_ratio - visible_ratio) / max(self.easy_ratio, 1e-6)
		hardness = max(0.0, min(1.0, hardness))
		return min_layers + int(round(span * hardness))

	def forward(self, inputs, mask = None, cost_mat = None, coords = None,
				raw_features = None):
		"""
		:param inputs:       ``(N, L, D)`` initial node embeddings
		:param mask:         ``(N, L)`` bool mask (True = hidden)
		:param cost_mat:     unused – kept for API compatibility
		:param coords:       ``(N, L, 2)`` fallback coordinates
		:param raw_features: ``(N, L, F)``  **preferred** – raw node features
		                     ``[x, y, demand, tw_start, tw_end, ...]``
		:return:             ``(N, L, D)`` encoded representations
		"""
		# resolve raw features
		if raw_features is not None:
			raw = raw_features
		elif coords is not None:
			raw = coords
		else:
			raw = inputs[:, :, :2]

		# structural encoding
		h = inputs + self.structural_encoding(raw)

		# multi-relational pairwise features
		edge_features = self.pairwise_features(raw)              # (N,L,L,6)

		# build attention mask
		attn_mask = None
		if self.k is not None and self.k > 0:
			dist = edge_features[..., 0]                         # normalised dist
			k = min(self.k, dist.size(-1))
			_, knn_idx = torch.topk(dist, k, largest = False)
			attn_mask = dist.new_ones(dist.size(), dtype = torch.bool)
			attn_mask.scatter_(2, knn_idx, False)
		if mask is not None:
			key_mask   = mask[:, None, :].expand(-1, edge_features.size(1), -1)
			attn_mask  = key_mask if attn_mask is None else (attn_mask | key_mask)
			query_mask = mask[:, :, None].expand(-1, -1, edge_features.size(2))
			attn_mask  = query_mask if attn_mask is None else (attn_mask | query_mask)

		# stacked VECTRA layers
		use_layers = self._resolve_layer_count(mask, len(self.layers))
		for layer in self.layers[:use_layers]:
			h = layer(h, edge_features,
					  attn_mask if attn_mask is not None else mask)
		return h


class GraphEncoderLayer(nn.Module):
	def __init__(self, head_count, model_size, ff_size, rbf_bins = 16, dropout = 0.1):
		super().__init__()
		self.head_count = head_count
		self.model_size = model_size
		self.rbf_bins = rbf_bins
		self.dropout = nn.Dropout(dropout)
		self.ff_dropout = nn.Dropout(dropout)
		if model_size % head_count != 0:
			raise ValueError("model_size must be divisible by head_count")

		head_dim = model_size // head_count
		self.q_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.k_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.v_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.out_proj = nn.Linear(head_count * head_dim, model_size, bias = False)

		self.edge_mlp = nn.Sequential(
			nn.Linear(rbf_bins, rbf_bins),
			nn.ReLU(),
			nn.Linear(rbf_bins, head_count)
		)

		self.norm1 = nn.LayerNorm(model_size)
		self.ff1 = nn.Linear(model_size, ff_size)
		self.ff2 = nn.Linear(ff_size, model_size)
		self.norm2 = nn.LayerNorm(model_size)

	def _rbf(self, dist):
		centers = torch.linspace(0, 1, self.rbf_bins, device = dist.device, dtype = dist.dtype)
		centers = centers.view(1, 1, 1, -1)
		width = (centers[..., 1] - centers[..., 0]).clamp(min = 1e-6)
		d = dist.unsqueeze(-1)
		return torch.exp(-((d - centers) ** 2) / (2 * width ** 2))

	def forward(self, h_in, cost_mat, mask = None):
		batch, length, _ = h_in.size()
		head_dim = self.model_size // self.head_count

		q = self.q_proj(h_in).view(batch, length, self.head_count, head_dim).transpose(1, 2)
		k = self.k_proj(h_in).view(batch, length, self.head_count, head_dim).transpose(1, 2)
		v = self.v_proj(h_in).view(batch, length, self.head_count, head_dim).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)

		rbf = self._rbf(cost_mat)
		edge_bias = self.edge_mlp(rbf).permute(0, 3, 1, 2)
		scores = scores + edge_bias

		if mask is not None:
			if mask.dim() == 2:
				key_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.head_count, length, -1)
			elif mask.dim() == 3:
				key_mask = mask.unsqueeze(1).expand(-1, self.head_count, -1, -1)
			else:
				raise ValueError("mask must be 2D or 3D")
			
			# Check if any row in any head is fully masked
			all_masked = key_mask.all(dim = -1, keepdim = True)
			if all_masked.any():
				# If all entries are masked, unmask the first one to avoid nan in softmax
				key_mask = key_mask.clone()
				key_mask.masked_fill_(all_masked, False)
				
			scores = scores.masked_fill(key_mask, float('-inf'))

		attn = F.softmax(scores, dim = -1)
		attn = self.dropout(attn)
		context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, length, -1)
		att = self.out_proj(context)

		h = self.norm1(h_in + att)
		ff = self.ff2(F.relu(self.ff1(h)))
		h_out = self.norm2(h + ff)
		if mask is not None:
			if mask.dim() == 2:
				h_out[mask] = 0
			elif mask.dim() == 3:
				node_mask = mask.all(dim = -1)
				if node_mask.any():
					h_out = h_out.masked_fill(node_mask.unsqueeze(-1), 0.0)
		return h_out


class GraphEncoder(nn.Module):
	def __init__(self, layer_count, head_count, model_size, ff_size, k = None,
			adaptive_depth = False, min_layers = 1, easy_ratio = 0.6):
		super().__init__()
		self.k = k
		self.adaptive_depth = adaptive_depth
		self.min_layers = max(1, min_layers)
		self.easy_ratio = easy_ratio
		self.layers = nn.ModuleList([
			GraphEncoderLayer(head_count, model_size, ff_size) for _ in range(layer_count)
		])

	def _resolve_layer_count(self, mask, total_layers):
		if (not self.adaptive_depth) or total_layers <= 1:
			return total_layers
		min_layers = min(self.min_layers, total_layers)
		if mask is None:
			return total_layers
		visible_ratio = 1.0 - mask.float().mean().item()
		if visible_ratio >= self.easy_ratio:
			return min_layers
		span = total_layers - min_layers
		hardness = (self.easy_ratio - visible_ratio) / max(self.easy_ratio, 1e-6)
		hardness = max(0.0, min(1.0, hardness))
		return min_layers + int(round(span * hardness))

	def forward(self, inputs, mask = None, cost_mat = None, coords = None):
		"""
		:param inputs: N x L_c x D
		:param mask:   N x L_c (optional)
		:param cost_mat: N x L_c x L_c (optional)
		:param coords: N x L_c x 2 (optional, raw positions)
		:return:       N x L_c x D
		"""
		if cost_mat is None:
			pos = coords if coords is not None else inputs[:, :, :2]
			cost_mat = torch.cdist(pos, pos)
		max_val = cost_mat.amax(dim = (-1, -2), keepdim = True).clamp(min = 1e-6)
		cost_mat = cost_mat / max_val
		attn_mask = None
		if self.k is not None and self.k > 0:
			k = min(self.k, cost_mat.size(-1))
			_, knn_idx = torch.topk(cost_mat, k, largest = False)
			attn_mask = cost_mat.new_ones(cost_mat.size(), dtype = torch.bool)
			attn_mask.scatter_(2, knn_idx, False)
		if mask is not None:
			key_mask = mask[:, None, :].expand(-1, cost_mat.size(1), -1)
			attn_mask = key_mask if attn_mask is None else (attn_mask | key_mask)
			query_mask = mask[:, :, None].expand(-1, -1, cost_mat.size(2))
			attn_mask = query_mask if attn_mask is None else (attn_mask | query_mask)
		h = inputs
		use_layers = self._resolve_layer_count(mask, len(self.layers))
		for layer in self.layers[:use_layers]:
			h = layer(h, cost_mat, attn_mask if attn_mask is not None else mask)
		return h


class FleetEncoderLayer(nn.Module):
	def __init__(self, head_count, model_size, ff_size):
		super().__init__()
		self.mha = _MHA_V2(head_count, model_size)
		self.norm1 = nn.LayerNorm(model_size)
		self.ff1 = nn.Linear(model_size, ff_size)
		self.ff2 = nn.Linear(ff_size, model_size)
		self.norm2 = nn.LayerNorm(model_size)

	def forward(self, veh_repr, cust_repr, mask = None):
		att = self.mha(veh_repr, keys = cust_repr, values = cust_repr, mask = mask)
		h = self.norm1(veh_repr + att)
		ff = self.ff2(F.relu(self.ff1(h)))
		h_out = self.norm2(h + ff)
		if mask is not None:
			if mask.dim() == 2 and mask.size(1) == h_out.size(1):
				h_out = h_out.masked_fill(mask.unsqueeze(-1), 0.0)
			elif mask.dim() == 3:
				node_mask = mask.all(dim = -1)
				if node_mask.any():
					h_out = h_out.masked_fill(node_mask.unsqueeze(-1), 0.0)
		return h_out


class FleetEncoder(nn.Module):
	def __init__(self, layer_count, head_count, model_size, ff_size,
			adaptive_depth = False, min_layers = 1, easy_ratio = 0.6):
		super().__init__()
		self.input_proj = nn.LazyLinear(model_size)
		self.adaptive_depth = adaptive_depth
		self.min_layers = max(1, min_layers)
		self.easy_ratio = easy_ratio
		self.layers = nn.ModuleList([
			FleetEncoderLayer(head_count, model_size, ff_size) for _ in range(layer_count)
		])

	def _resolve_layer_count(self, mask, total_layers):
		if (not self.adaptive_depth) or total_layers <= 1:
			return total_layers
		min_layers = min(self.min_layers, total_layers)
		if mask is None:
			return total_layers
		if mask.dim() == 3:
			feasible_ratio = (~mask).float().mean().item()
		elif mask.dim() == 2:
			feasible_ratio = (~mask).float().mean().item()
		else:
			return total_layers
		if feasible_ratio >= self.easy_ratio:
			return min_layers
		span = total_layers - min_layers
		hardness = (self.easy_ratio - feasible_ratio) / max(self.easy_ratio, 1e-6)
		hardness = max(0.0, min(1.0, hardness))
		return min_layers + int(round(span * hardness))

	def forward(self, vehicles, cust_repr, mask = None):
		"""
		:param vehicles:    N x L_v x D_v
		:param cust_repr:   N x L_c x D
		:param mask:        N x L_v x L_c or N x L_c (optional)
		:return:            N x L_v x D
		"""
		h = self.input_proj(vehicles)
		use_layers = self._resolve_layer_count(mask, len(self.layers))
		for layer in self.layers[:use_layers]:
			h = layer(h, cust_repr, mask)
		return h


class EdgeFeatureEncoder(nn.Module):
	def __init__(self, edge_feat_size, model_size):
		super().__init__()
		self.edge_feat_size = edge_feat_size
		self.model_size = model_size
		self.net = nn.Sequential(
			nn.Linear(edge_feat_size, model_size),
			nn.ReLU(),
			nn.Linear(model_size, model_size),
			nn.LayerNorm(model_size)
		)

	def forward(self, edge_feat):
		"""
		:param edge_feat: N x 1 x L_c x E
		:return:          N x 1 x L_c x D
		"""
		return self.net(edge_feat)


class CrossEdgeFusion(nn.Module):
	def __init__(self, head_count, model_size):
		super().__init__()
		self.head_count = head_count
		self.model_size = model_size
		if model_size % head_count != 0:
			raise ValueError("model_size must be divisible by head_count")
		head_dim = model_size // head_count
		self.q_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.k_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.edge_bias = nn.Linear(model_size, head_count, bias = False)

	def forward(self, veh_repr, cust_repr, edge_emb):
		"""
		:param veh_repr: N x 1 x D
		:param cust_repr: N x L_c x D
		:param edge_emb: N x 1 x L_c x D
		:return:         N x 1 x L_c
		"""
		batch, _, dim = veh_repr.size()
		_, length, _ = cust_repr.size()
		head_dim = dim // self.head_count

		q = self.q_proj(veh_repr).view(batch, 1, self.head_count, head_dim).transpose(1, 2)
		k = self.k_proj(cust_repr).view(batch, length, self.head_count, head_dim).transpose(1, 2)
		scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)

		bias = self.edge_bias(edge_emb).permute(0, 3, 1, 2)
		scores = scores + bias

		return scores.mean(dim = 1)


class CoordinationMemory(nn.Module):
	def __init__(self, veh_state_size, hidden_size):
		super().__init__()
		self.veh_state_size = veh_state_size
		self.hidden_size = hidden_size
		self.input_proj = nn.LazyLinear(hidden_size)
		self.hidden_proj = nn.Linear(hidden_size, hidden_size)

	def update(self, memory, veh_idx, veh_repr, cust_repr, edge_emb):
		"""
		:param memory:   N x L_v x H
		:param veh_idx:  N x 1
		:param veh_repr: N x 1 x D
		:param cust_repr: N x 1 x D
		:param edge_emb: N x 1 x 1 x D
		:return:         N x L_v x H (updated)
		"""
		cur_h = memory.gather(1, veh_idx[:, :, None].expand(-1, -1, memory.size(-1)))
		x = torch.cat((
			veh_repr.squeeze(1),
			cust_repr.squeeze(1),
			edge_emb.squeeze(2).squeeze(1)
		), dim = -1)
		next_h = torch.tanh(self.input_proj(x) + self.hidden_proj(cur_h.squeeze(1)))
		next_h = next_h.to(memory.dtype)
		updated = memory.scatter(1, veh_idx[:, :, None].expand(-1, -1, memory.size(-1)), next_h.unsqueeze(1))
		return updated


class OwnershipHead(nn.Module):
	def __init__(self, model_size):
		super().__init__()
		self.model_size = model_size
		self.veh_proj = nn.LazyLinear(model_size)
		self.cust_proj = nn.Linear(model_size, model_size, bias = False)

	def forward(self, veh_memory, cust_repr):
		"""
		:param veh_memory: N x L_v x H
		:param cust_repr:  N x L_c x D
		:return:           N x L_v x L_c
		"""
		v = self.veh_proj(veh_memory)
		c = self.cust_proj(cust_repr)
		logits = torch.matmul(v, c.transpose(1, 2))
		logits *= (self.model_size ** -0.5)
		return logits


class LookaheadHead(nn.Module):
	def __init__(self, model_size, hidden_size = 128, dropout = 0.1):
		super().__init__()
		self.model_size = model_size
		self.hidden_size = hidden_size
		self.net = nn.Sequential(
			nn.LazyLinear(hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, 1)
		)

	def forward(self, veh_repr, cust_repr, edge_emb):
		"""
		:param veh_repr: N x 1 x D
		:param cust_repr: N x L_c x D
		:param edge_emb: N x 1 x L_c x D
		:return:         N x 1 x L_c
		"""
		batch, _, dim = veh_repr.size()
		length = cust_repr.size(1)
		veh_expand = veh_repr.expand(-1, length, -1)
		edge_flat = edge_emb.squeeze(1)
		feat = torch.cat((veh_expand, cust_repr, edge_flat), dim = -1)
		out = self.net(feat).transpose(1, 2)
		return out


# ============================================================================
# Situation-Aware MoE Score Fusion
# ============================================================================
#
# Theoretical Motivation
# ----------------------
# In DVRPTW, the decoder must fuse multiple scoring signals to select the
# next customer: cross-attention compatibility, fleet-level ownership
# coordination, and lookahead cost estimation.  A static MLP fusion
# (the baseline) applies IDENTICAL weights regardless of the decision
# context.  This is sub-optimal because the relative informativeness of
# each signal varies dramatically:
#
#   (a) High remaining capacity + wide TW → spatial proximity dominates.
#       The attention signal is most reliable; ownership/lookahead noise hurts.
#   (b) Tight capacity or narrow TW → feasibility constraints dominate.
#       Lookahead and ownership carry critical information.
#   (c) Multiple vehicles competing for the same customers → coordination
#       (ownership) must override pure greedy proximity.
#   (d) Near end of planning horizon → return-to-depot lookahead becomes
#       decisive.
#
# A Mixture-of-Experts (MoE) architecture (Shazeer et al., 2017; Fedus
# et al., 2022) naturally handles this: K expert networks each encode a
# different fusion "strategy", and a context-conditioned gating network
# dynamically selects which strategy to apply.
#
# Key innovation: **Dual-Granularity Gating**
#   1. *Situation gate*  – operates on the vehicle's compressed state +
#      graph summary → captures global context (route progress,
#      remaining capacity, fleet state).
#   2. *Candidate gate*  – operates on each candidate's scoring pattern +
#      edge context → captures per-candidate context (whether this
#      candidate is more proximity-driven or urgency-driven).
#   The two gate levels are combined via a learnable sigmoid-balanced mixture.
#
# Additional contributions:
#   •  Batched expert computation (single einsum instead of K forward passes)
#   •  Diversity-aware initialisation (expert k biased toward score k)
#   •  Load-balancing auxiliary loss for stable expert utilisation
#   •  Edge-feature enriched candidate gating
# ============================================================================


class AdaptiveExpertScoreFusion(nn.Module):
	"""Situation-Aware Mixture-of-Experts score fusion for DVRPTW decoding.

	Given K scoring signals (e.g. attention, ownership, lookahead), this
	module learns to fuse them differently depending on the current
	decision context, using a soft MoE architecture.

	Parameters
	----------
	num_experts : int
		Number of expert fusion networks (default: 4).
	model_size : int
		Dimension D of vehicle / customer representations.
	num_scores : int
		Number of input score channels (default: 3).
	expert_hidden : int
		Hidden dimension inside each expert MLP.
	gate_hidden : int
		Hidden dimension of the situation gating MLP.
	gate_noise : float
		Std-dev of Gaussian noise added to gate logits during training
		to encourage expert exploration.
	dropout : float
		Dropout rate on gate weights.
	"""

	def __init__(self, num_experts = 4, model_size = 128, num_scores = 3,
				 expert_hidden = 32, gate_hidden = 64, gate_noise = 0.1,
				 dropout = 0.1):
		super().__init__()
		self.num_experts   = num_experts
		self.num_scores    = num_scores
		self.model_size    = model_size
		self.gate_noise    = gate_noise

		# === Expert networks (batched as tensor ops for efficiency) ===
		# Each expert: score_vec (S) → hidden (H) → scalar (1)
		self.expert_w1 = nn.Parameter(torch.empty(num_experts, num_scores, expert_hidden))
		self.expert_b1 = nn.Parameter(torch.zeros(num_experts, expert_hidden))
		self.expert_w2 = nn.Parameter(torch.empty(num_experts, expert_hidden, 1))
		self.expert_b2 = nn.Parameter(torch.zeros(num_experts, 1))
		self._init_experts()

		# === Situation gate ===
		# Input: vehicle repr (D) + graph summary (D)  → K expert logits
		self.situation_gate = nn.Sequential(
			nn.Linear(model_size * 2, gate_hidden),
			nn.GELU(),
			nn.Linear(gate_hidden, num_experts),
		)

		# === Candidate gate ===
		# Input: per-candidate score pattern (S)  → K expert logits
		self.candidate_gate = nn.Linear(num_scores, num_experts)

		# === Edge-context enrichment for candidate gate ===
		# Projects rich edge embedding to lightweight gate contribution
		self.edge_gate = nn.Linear(model_size, num_experts, bias = False)

		# === Learnable balance between situation-level and candidate-level ===
		# sigmoid(0) = 0.5  →  equal initial weight
		self.gate_balance = nn.Parameter(torch.tensor(0.0))

		self.gate_dropout = nn.Dropout(dropout)

		# Auxiliary loss and analysis storage
		self._aux_loss = None
		self._last_gate_weights = None

	# ------------------------------------------------------------------

	def _init_experts(self):
		"""Kaiming init with diversity: expert k is biased toward score k."""
		K, S, H = self.expert_w1.shape
		for k in range(K):
			nn.init.kaiming_uniform_(self.expert_w1.data[k], a = math.sqrt(5))
			nn.init.kaiming_uniform_(self.expert_w2.data[k], a = math.sqrt(5))
		# Bias expert k toward score source k (modular wrap for K > S)
		with torch.no_grad():
			for k in range(K):
				if k < S:
					self.expert_w1.data[k, k, :] += 0.5

	# ------------------------------------------------------------------

	def forward(self, scores, veh_repr, cust_repr, edge_emb = None):
		"""
		:param scores:    ``(N, 1, L_c, S)`` z-normalized score stack
		:param veh_repr:  ``(N, 1, D)`` current vehicle representation
		:param cust_repr: ``(N, L_c, D)`` customer representations
		:param edge_emb:  ``(N, 1, L_c, D)`` edge embeddings (optional)
		:return:          ``(N, 1, L_c)`` fused compatibility scores
		"""
		N, _, L, S = scores.shape

		# ---- Situation-level gating (global context) ----
		graph_ctx = cust_repr.mean(dim = 1, keepdim = True)          # (N, 1, D)
		sit_input = torch.cat([veh_repr, graph_ctx], dim = -1)       # (N, 1, 2D)
		sit_logits = self.situation_gate(sit_input)                   # (N, 1, K)

		# ---- Candidate-level gating (per-candidate context) ----
		cand_logits = self.candidate_gate(scores)                     # (N, 1, L, K)
		if edge_emb is not None:
			cand_logits = cand_logits + self.edge_gate(edge_emb)      # (N, 1, L, K)

		# ---- Combine the two gate levels ----
		alpha = torch.sigmoid(self.gate_balance)
		gate_logits = alpha * sit_logits.unsqueeze(2) + (1 - alpha) * cand_logits
		#  → (N, 1, L, K)

		# Exploration noise during training
		if self.training and self.gate_noise > 0:
			gate_logits = gate_logits + torch.randn_like(gate_logits) * self.gate_noise

		gate_weights = F.softmax(gate_logits, dim = -1)               # (N, 1, L, K)
		gate_weights = self.gate_dropout(gate_weights)

		# ---- Batched expert computation ----
		# scores: (..., S) × expert_w1: (K, S, H) → (..., K, H)
		h = torch.einsum('...s, ksh -> ...kh', scores, self.expert_w1) + self.expert_b1
		h = F.gelu(h)
		expert_out = torch.einsum('...kh, kho -> ...ko', h, self.expert_w2) + self.expert_b2
		expert_out = expert_out.squeeze(-1)                           # (N, 1, L, K)

		# ---- Weighted combination ----
		fused = (gate_weights * expert_out).sum(dim = -1)             # (N, 1, L)

		# ---- Auxiliary load-balancing loss (MoE standard) ----
		if self.training:
			# fraction: actual routing fraction per expert (detached)
			fraction = gate_weights.detach().mean(dim = (0, 1, 2))    # (K,)
			# prob: mean gate probability per expert (with gradient)
			prob = F.softmax(gate_logits, dim = -1).mean(dim = (0, 1, 2))
			self._aux_loss = self.num_experts * (fraction * prob).sum()
		else:
			self._aux_loss = scores.new_tensor(0.0)

		# Store for analysis / visualisation
		self._last_gate_weights = gate_weights.detach()

		return fused

	# ------------------------------------------------------------------

	@torch.no_grad()
	def get_expert_utilization(self):
		"""Return per-expert mean gate weight from the last forward pass.

		Useful for monitoring expert collapse during training.

		:return: ``(K,)`` tensor or None if no forward pass yet
		"""
		if self._last_gate_weights is None:
			return None
		return self._last_gate_weights.mean(dim = (0, 1, 2))

