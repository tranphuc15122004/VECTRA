# VECTRA: Vehicle-Edge-Customer Temporal Routing Anticipation

**Dynamic Vehicle Routing with Multi-Agent Deep Reinforcement Learning**

## Tổng quan

Repository này triển khai một kiến trúc học sâu tiên tiến cho bài toán **Dynamic Vehicle Routing Problem with Time Windows (DVRPTW)** sử dụng **Edge-Enhanced Learning** kết hợp với **Coordination Memory** và **Adaptive Planning**.

### Đặc điểm chính

Model được thiết kế theo paradigm **sequential Multi-agent Markov Decision Process (sMMDP)**:
- **Shared policy**: Một policy duy nhất được chia sẻ cho tất cả các phương tiện
- **Sequential decision**: Tại mỗi bước, chỉ có một phương tiện đưa ra quyết định
- **Per-vehicle memory**: Mỗi phương tiện duy trì trạng thái ẩn riêng để phối hợp với đội xe
- **Edge-aware representation**: Biểu diễn chi tiết các ràng buộc giữa xe và khách hàng

### Kiến trúc tổng thể

Model kết hợp 7 thành phần chính:

1. **GraphEncoder**: Mã hóa quan hệ không gian giữa các khách hàng với RBF distance bias
2. **FleetEncoder**: Mã hóa tương tác giữa các phương tiện trong đội xe
3. **EdgeFeatureEncoder**: Mã hóa đặc trưng edge (xe-khách hàng) với 8 features
4. **CrossEdgeFusion**: Tính attention score có điều chỉnh edge bias
5. **CoordinationMemory**: Bộ nhớ phối hợp tránh xung đột giữa các xe
6. **OwnershipHead**: Dự đoán xe nào nên phục vụ khách hàng nào
7. **LookaheadHead**: Ước lượng chi phí tương lai để tránh quyết định tham lam

### Quy trình ra quyết định

Tại mỗi bước, xe đang hoạt động chọn khách hàng tiếp theo thông qua:

$$\text{score}_j = \text{ScoreFusion}(\text{att\_score}_j, \text{owner\_bias}_j, \text{lookahead}_j)$$

Trong đó **ScoreFusion** là MLP học cách kết hợp 3 nguồn tín hiệu:
- **att_score**: Độ tương thích attention giữa xe và khách hàng
- **owner_bias**: Xác suất xe này nên phục vụ khách hàng (log-prob)
- **lookahead**: Ước lượng chi phí tương lai nếu chọn khách hàng này

---

## Chi tiết các thành phần

### 1. GraphEncoder - Mã hóa đồ thị khách hàng

**Chức năng**: Mã hóa cấu trúc không gian của các khách hàng sử dụng graph attention với RBF distance bias.

**Kiến trúc**:
```
GraphEncoder
├── GraphEncoderLayer × layer_count (default 3)
│   ├── Multi-Head Attention với RBF edge bias
│   │   ├── Q, K, V projections (D → heads × head_dim)
│   │   ├── RBF(dist) → edge_mlp → bias cho mỗi head
│   │   └── Attention mask: k-NN sparsification
│   ├── LayerNorm + Residual
│   ├── Feed-Forward (D → ff_size → D)
│   └── LayerNorm + Residual
```

**Inputs**:
- `customers`: $N \times L_c \times D_c$ - Node features (depot ở index 0)
  - $N$: Batch size
  - $L_c$: Số lượng nodes (depot + customers)
  - $D_c$: Số chiều features (x, y, demand, tw_start, tw_end, ...)
- `mask`: $N \times L_c$ - Binary mask (1 = hidden/padding)
- `coords`: $N \times L_c \times 2$ - Tọa độ (x, y)

**Outputs**:
- `cust_repr`: $N \times L_c \times D$ - Encoded representations
  - $D$: model_size (default 128)

**Chi tiết xử lý**:

1. **Embedding**:
   ```python
   depot_emb = depot_embedding(customers[:, 0:1, :])     # N × 1 × D
   cust_emb = cust_embedding(customers[:, 1:, :])        # N × (L_c-1) × D
   h = torch.cat([depot_emb, cust_emb], dim=1)           # N × L_c × D
   ```

2. **Distance matrix & normalization**:
   ```python
   dist = torch.cdist(coords, coords)                    # N × L_c × L_c
   max_dist = dist.amax(dim=(-1, -2), keepdim=True)
   cost_mat = dist / max_dist.clamp(min=1e-6)            # Normalize to [0, 1]
   ```

3. **k-NN sparsification** (nếu k > 0):
   ```python
   _, knn_idx = torch.topk(cost_mat, k, largest=False)   # Top-k closest
   attn_mask = torch.ones_like(cost_mat, dtype=bool)
   attn_mask.scatter_(2, knn_idx, False)                 # Only attend to k neighbors
   ```

4. **RBF distance encoding**:
   ```python
   # Chia [0, 1] thành rbf_bins centers (default 16)
   centers = torch.linspace(0, 1, rbf_bins)
   width = centers[1] - centers[0]
   RBF = exp(-((cost_mat - centers)^2) / (2 * width^2))  # N × L_c × L_c × rbf_bins
   ```

5. **Edge-biased attention**:
   ```python
   # Standard attention
   Q = q_proj(h).view(N, L_c, heads, head_dim)
   K = k_proj(h).view(N, L_c, heads, head_dim)
   V = v_proj(h).view(N, L_c, heads, head_dim)
   
   scores = (Q @ K^T) / sqrt(head_dim)                   # N × heads × L_c × L_c
   
   # Edge bias
   edge_bias = edge_mlp(RBF)                             # N × L_c × L_c × heads
   edge_bias = edge_bias.permute(0, 3, 1, 2)            # N × heads × L_c × L_c
   
   scores = scores + edge_bias                           # Add bias
   scores = scores.masked_fill(attn_mask, -inf)         # Apply k-NN mask
   
   attn = softmax(scores, dim=-1)
   context = (attn @ V)                                  # N × heads × L_c × head_dim
   ```

6. **Multi-layer encoding**:
   - Mỗi layer: `h = LayerNorm(h + Attention(h)) → LayerNorm(h + FFN(h))`
   - Masked nodes được zero ra sau mỗi layer

**Ý nghĩa**:
- **RBF bias**: Model học được các mẫu liên quan đến khoảng cách (gần → có thể cluster, xa → có thể bỏ qua)
- **k-NN sparsification**: Giảm complexity từ $O(L_c^2)$ → $O(L_c \cdot k)$, tập trung vào local structure
- **Multi-layer**: Lan truyền thông tin qua nhiều bước → mỗi node "biết" về global context

---

### 2. FleetEncoder - Mã hóa đội xe

**Chức năng**: Mã hóa tương tác giữa các phương tiện và khách hàng.

**Kiến trúc**:
```
FleetEncoder
├── input_proj: LazyLinear(veh_state_size → model_size)
└── FleetEncoderLayer × layer_count (default 3)
    ├── Cross-Attention (vehicles attend to customers)
    │   └── _MHA_V2(head_count, model_size)
    ├── LayerNorm + Residual
    ├── Feed-Forward (D → ff_size → D)
    └── LayerNorm + Residual
```

**Inputs**:
- `vehicles`: $N \times L_v \times D_v$ - Vehicle states
  - $L_v$: Số phương tiện
  - $D_v$: State size (x, y, capacity, current_time, ...)
- `cust_repr`: $N \times L_c \times D$ - Customer representations từ GraphEncoder
- `mask`: $N \times L_v \times L_c$ hoặc $N \times L_c$ - Feasibility mask

**Outputs**:
- `fleet_repr`: $N \times L_v \times D$ - Vehicle representations

**Chi tiết xử lý**:

1. **Projection**: 
   ```python
   h = input_proj(vehicles)  # N × L_v × D
   ```

2. **Cross-attention** (mỗi layer):
   ```python
   # Query from vehicles, Key & Value from customers
   Q = q_proj(h)                          # N × L_v × D
   K = k_proj(cust_repr)                  # N × L_c × D
   V = v_proj(cust_repr)                  # N × L_c × D
   
   scores = (Q @ K^T) / sqrt(D)           # N × L_v × L_c
   if mask is not None:
       scores = scores.masked_fill(mask, -inf)
   
   attn = softmax(scores, dim=-1)
   context = attn @ V                     # N × L_v × D
   
   h = LayerNorm(h + context)
   h = LayerNorm(h + FFN(h))
   ```

3. **Multi-layer refinement**:
   - Mỗi layer làm giàu vehicle repr với context từ customers
   - Residual connections duy trì thông tin vehicle state ban đầu

**Ý nghĩa**:
- Vehicle biết được "cơ hội" xung quanh nó (customers nào gần, feasible)
- Cross-attention cho phép vehicle "nhìn" toàn bộ customers
- Mask đảm bảo chỉ xem xét customers khả thi (time window, capacity)

---

### 3. EdgeFeatureEncoder - Mã hóa đặc trưng edge

**Chức năng**: Mã hóa các ràng buộc chi tiết giữa xe đang hoạt động và từng khách hàng.

**Kiến trúc**:
```
EdgeFeatureEncoder (MLP)
├── Linear(edge_feat_size → model_size)
├── ReLU
├── Linear(model_size → model_size)
└── LayerNorm
```

**Edge features** ($E = 8$ dimensions):

| Feature | Formula | Ý nghĩa |
|---------|---------|---------|
| **distance** | $\|pos_{veh} - pos_{cust}\|_2$ | Khoảng cách Euclidean |
| **travel_time** | $\text{distance} / \text{speed}$ | Thời gian di chuyển |
| **arrival** | $t_{veh} + \text{travel\_time}$ | Thời điểm đến |
| **wait** | $\max(0, tw_{start} - \text{arrival})$ | Thời gian chờ đợi |
| **late** | $\max(0, \text{arrival} - tw_{end})$ | Trễ hẹn (vi phạm) |
| **slack** | $tw_{end} - \text{arrival}$ | Thời gian dư (có thể âm) |
| **feasible** | $1.0$ if valid, $0.0$ if masked | Khả thi |
| **cap_gap** | $\text{capacity}_{veh} - \text{demand}_{cust}$ | Dư thừa capacity |

**Inputs**:
- `edge_feat`: $N \times 1 \times L_c \times E$ - Raw edge features

**Outputs**:
- `edge_emb`: $N \times 1 \times L_c \times D$ - Encoded edge features

**Cách tính edge features** (trong `_build_edge_features`):

```python
def _build_edge_features(vehicles, customers, veh_idx):
    # 1. Gather acting vehicle state
    v = vehicles.gather(1, veh_idx[:, :, None].expand(-1, -1, D_v))
    v_pos = v[:, :, :2]       # (x, y)
    v_time = v[:, :, 3:4]     # current time
    v_capa = v[:, :, 2:3]     # remaining capacity
    
    # 2. Customer features
    c_pos = customers[:, :, :2]
    c_demand = customers[:, :, 2:3]
    c_tw_start = customers[:, :, 3:4]
    c_tw_end = customers[:, :, 4:5]
    
    # 3. Compute edge features
    dist = torch.cdist(v_pos, c_pos)                    # N × 1 × L_c
    travel_time = dist / speed                          # Assume speed = 1.0
    arrival = v_time.expand_as(travel_time) + travel_time
    
    wait = (c_tw_start.T - arrival).clamp(min=0)
    late = (arrival - c_tw_end.T).clamp(min=0)
    slack = c_tw_end.T - arrival
    
    cap_gap = v_capa.expand_as(c_demand.T) - c_demand.T
    feasible = (~veh_mask).float() if veh_mask else ones_like(dist)
    
    # 4. Stack into tensor
    edge_feat = torch.cat([
        dist.unsqueeze(-1),
        travel_time.unsqueeze(-1),
        arrival.unsqueeze(-1),
        wait.unsqueeze(-1),
        late.unsqueeze(-1),
        slack.unsqueeze(-1),
        feasible.unsqueeze(-1),
        cap_gap.unsqueeze(-1)
    ], dim=-1)  # N × 1 × L_c × 8
    
    return edge_feat
```

**Processing**:
```python
edge_emb = net(edge_feat)  # MLP: Linear → ReLU → Linear → LayerNorm
```

**Ý nghĩa**:
- **Encode rõ ràng các ràng buộc cứng**: time windows, capacity
- **Model học được penalty** cho late arrival, reward cho slack time
- **Feasibility feature** giúp soft mask thay vì hard mask
- **Edge features bổ sung** cho attention scores → quyết định tốt hơn

---

### 4. CrossEdgeFusion - Attention với edge bias

**Chức năng**: Tính compatibility score giữa xe và khách hàng với edge-aware attention.

**Kiến trúc**:
```
CrossEdgeFusion
├── q_proj: Linear(D → head_count × head_dim)
├── k_proj: Linear(D → head_count × head_dim)
└── edge_bias: Linear(D → head_count)
```

**Inputs**:
- `veh_repr`: $N \times 1 \times D$ - Acting vehicle representation
- `cust_repr`: $N \times L_c \times D$ - All customers representations
- `edge_emb`: $N \times 1 \times L_c \times D$ - Edge embeddings

**Outputs**:
- `att_score`: $N \times 1 \times L_c$ - Attention scores

**Chi tiết xử lý**:

```python
def forward(veh_repr, cust_repr, edge_emb):
    N, _, D = veh_repr.size()
    L_c = cust_repr.size(1)
    head_dim = D // head_count
    
    # 1. Multi-head projections
    Q = q_proj(veh_repr)  # N × 1 × (heads × head_dim)
    Q = Q.view(N, 1, heads, head_dim).transpose(1, 2)  # N × heads × 1 × head_dim
    
    K = k_proj(cust_repr)  # N × L_c × (heads × head_dim)
    K = K.view(N, L_c, heads, head_dim).transpose(1, 2)  # N × heads × L_c × head_dim
    
    # 2. Attention scores
    scores = torch.matmul(Q, K.transpose(-1, -2))  # N × heads × 1 × L_c
    scores = scores / math.sqrt(head_dim)
    
    # 3. Edge bias
    bias = edge_bias(edge_emb)  # N × 1 × L_c × heads
    bias = bias.permute(0, 3, 1, 2)  # N × heads × 1 × L_c
    
    scores = scores + bias
    
    # 4. Aggregate across heads
    att_score = scores.mean(dim=1)  # N × 1 × L_c
    
    return att_score
```

**Ý nghĩa**:
- **Edge bias điều chỉnh attention** dựa trên constraints (time, capacity, distance)
- **Mỗi head học một khía cạnh** khác nhau của compatibility
  - Head 1: proximity
  - Head 2: time window urgency
  - Head 3: capacity fit
  - ...
- **Mean over heads** tạo robust final score

---

### 5. CoordinationMemory - Bộ nhớ phối hợp

**Chức năng**: Duy trì hidden state cho mỗi phương tiện để phối hợp quyết định và tránh xung đột.

**Kiến trúc**:
```
CoordinationMemory (GRU-like update)
├── input_proj: LazyLinear(3D → hidden_size)
└── hidden_proj: Linear(hidden_size → hidden_size)
```

**State**:
- `memory`: $N \times L_v \times H$ - Hidden states cho tất cả vehicles
  - $H$: hidden_size (mặc định bằng model_size)

**Update mechanism** (tương tự GRU):

```python
def update(memory, veh_idx, veh_repr, cust_repr, edge_emb):
    """
    Cập nhật memory chỉ cho xe đang hoạt động
    
    Args:
        memory: N × L_v × H - current memory states
        veh_idx: N × 1 - index of acting vehicle
        veh_repr: N × 1 × D - vehicle representation
        cust_repr: N × 1 × D - selected customer representation
        edge_emb: N × 1 × 1 × D - edge embedding of (vehicle, customer)
    
    Returns:
        updated_memory: N × L_v × H
    """
    # 1. Gather current hidden state của xe đang hoạt động
    cur_h = memory.gather(1, veh_idx[:, :, None].expand(-1, -1, H))
    # Shape: N × 1 × H
    
    # 2. Concatenate context
    veh = veh_repr.squeeze(1)              # N × D
    cust = cust_repr.squeeze(1)            # N × D
    edge = edge_emb.squeeze(2).squeeze(1)  # N × D
    x = torch.cat([veh, cust, edge], dim=-1)  # N × 3D
    
    # 3. Compute new hidden state (GRU-like)
    next_h = torch.tanh(
        input_proj(x) + hidden_proj(cur_h.squeeze(1))
    )  # N × H
    
    # 4. Scatter update (chỉ cập nhật acting vehicle slot)
    next_h = next_h.unsqueeze(1)  # N × 1 × H
    updated_memory = memory.scatter(
        1, 
        veh_idx[:, :, None].expand(-1, -1, H), 
        next_h
    )
    
    return updated_memory
```

**Timeline của memory**:

```
Step 0:  memory = [0, 0, 0, 0, 0]  (all vehicles initialized to zero)

Step 1:  veh_idx = 2, chọn customer 5
         memory = [0, 0, h1, 0, 0]  (chỉ vehicle 2 được update)

Step 2:  veh_idx = 0, chọn customer 3
         memory = [h2, 0, h1, 0, 0]  (vehicle 0 được update)

Step 3:  veh_idx = 2, chọn customer 7
         memory = [h2, 0, h3, 0, 0]  (vehicle 2 được update lại)

...
```

**Ý nghĩa**:
- **Mỗi xe "nhớ" lịch sử hành động** của nó
- **Memory được dùng bởi OwnershipHead** để dự đoán phân công
- **Tránh xung đột**: 
  - Nếu vehicle A đã "claim" customers ở khu vực Đông, memory của A phản ánh điều này
  - OwnershipHead học được → dự đoán customers Đông nên thuộc A
  - Vehicle B (đang ở Tây) sẽ có owner_bias thấp cho customers Đông → ít chọn
- **Coordination implicitly**: Không cần communication trực tiếp, học qua RL

---

### 6. OwnershipHead - Phân công khách hàng

**Chức năng**: Dự đoán vehicle nào phù hợp nhất để phục vụ mỗi customer (soft assignment).

**Kiến trúc**:
```
OwnershipHead
├── veh_proj: LazyLinear(H → D)
└── cust_proj: Linear(D → D, bias=False)
```

**Inputs**:
- `veh_memory`: $N \times L_v \times H$ - Hidden states của tất cả vehicles
- `cust_repr`: $N \times L_c \times D$ - Customer representations

**Outputs**:
- `owner_logits`: $N \times L_v \times L_c$ - Logits cho mỗi cặp (vehicle, customer)

**Chi tiết xử lý**:

```python
def forward(veh_memory, cust_repr):
    """
    Compute ownership scores
    
    Returns:
        owner_logits: N × L_v × L_c
    """
    # 1. Project to common space
    v = veh_proj(veh_memory)   # N × L_v × D
    c = cust_proj(cust_repr)   # N × L_c × D
    
    # 2. Dot-product affinity matrix
    logits = torch.matmul(v, c.transpose(1, 2))  # N × L_v × L_c
    logits = logits / math.sqrt(D)  # Scale by sqrt(D)
    
    return logits
```

**Sử dụng trong step()**:

```python
# 1. Compute ownership probabilities
owner_logits = ownership_head(veh_memory, cust_repr)  # N × L_v × L_c
owner_prob = F.softmax(owner_logits, dim=1)           # Softmax over vehicles

# 2. Extract for acting vehicle
owner_prob_acting = owner_prob.gather(
    1, 
    veh_idx[:, :, None].expand(-1, -1, L_c)
)  # N × 1 × L_c

# 3. Convert to log-prob bias
owner_bias = owner_prob_acting.clamp(min=1e-9).log()  # N × 1 × L_c
```

**Ví dụ cụ thể**:

```
Scenario: 3 vehicles, 5 customers

owner_logits = [
    [2.0, 1.0, -1.0, 0.5, 1.5],   # Vehicle 0
    [0.5, 2.5,  1.0, 2.0, 0.0],   # Vehicle 1
    [1.0, 0.0,  3.0, 1.0, 2.0],   # Vehicle 2
]

After softmax over vehicles (dim=1):
owner_prob = [
    [0.50, 0.21, 0.05, 0.23, 0.36],  # Prob vehicle 0 should serve each customer
    [0.11, 0.69, 0.21, 0.61, 0.08],  # Prob vehicle 1 should serve each customer
    [0.39, 0.10, 0.74, 0.16, 0.56],  # Prob vehicle 2 should serve each customer
]

Nếu acting vehicle là veh_idx = 1:
owner_bias = log([0.11, 0.69, 0.21, 0.61, 0.08])
           = [-2.21, -0.37, -1.56, -0.49, -2.53]

→ Customer 1 có owner_bias cao nhất → vehicle 1 nên ưu tiên chọn customer 1
→ Customer 4 cũng cao → ưu tiên thứ hai
→ Customer 0 và 4 có owner_bias thấp → nên tránh (để cho vehicle 0 và 2)
```

**Ý nghĩa**:
- **Soft assignment**: Không cứng nhắc, chỉ là bias
- **Fleet coordination**: Các xe "tự phân chia" customers
- **Learned từ RL**: Model học được cách phân chia tốt thông qua reward signal
- **Giảm conflicts**: Nếu vehicle A có high ownership cho customer j, các xe khác sẽ ít chọn j

---

### 7. LookaheadHead - Ước lượng chi phí tương lai

**Chức năng**: Estimate future cost-to-go (value function) để tránh greedy decisions.

**Kiến trúc**:
```
LookaheadHead (Value network)
├── LazyLinear(3D → hidden_size)
├── ReLU
├── Dropout(0.1)
└── Linear(hidden_size → 1)
```

**Inputs**:
- `veh_repr`: $N \times 1 \times D$ - Acting vehicle
- `cust_repr`: $N \times L_c \times D$ - All customers
- `edge_emb`: $N \times 1 \times L_c \times D$ - Edge features

**Outputs**:
- `lookahead`: $N \times 1 \times L_c$ - Estimated future cost for each customer

**Chi tiết xử lý**:

```python
def forward(veh_repr, cust_repr, edge_emb):
    """
    Estimate V(s') for each possible next customer
    
    V(s') ≈ "Nếu chọn customer j, chi phí từ sau đó đến hết episode là bao nhiêu?"
    """
    N, _, D = veh_repr.size()
    L_c = cust_repr.size(1)
    
    # 1. Expand vehicle repr to match all customers
    veh_expand = veh_repr.expand(-1, L_c, -1)  # N × L_c × D
    
    # 2. Flatten edge embeddings
    edge_flat = edge_emb.squeeze(1)  # N × L_c × D
    
    # 3. Concatenate context for each customer
    feat = torch.cat([
        veh_expand,   # Vehicle state
        cust_repr,    # Customer features
        edge_flat     # Edge constraints
    ], dim=-1)  # N × L_c × 3D
    
    # 4. MLP prediction
    lookahead = net(feat)  # N × L_c × 1
    lookahead = lookahead.transpose(1, 2)  # N × 1 × L_c
    
    return lookahead
```

**Ví dụ cụ thể**:

```
Scenario: Vehicle ở (0, 0), còn 3 customers

Customer A: (1, 1), gần, nhưng isolated
Customer B: (2, 0), vừa, nằm giữa 2 clusters
Customer C: (10, 10), xa, nhưng trong cluster lớn

Lookahead estimates:
lookahead_A = 50  (cao vì sau khi đi A phải quay lại, detour lớn)
lookahead_B = 30  (trung bình, vị trí tốt)
lookahead_C = 20  (thấp vì sau khi đến C, có nhiều customers gần đó)

→ Nếu chỉ xem att_score (distance), sẽ chọn A (gần nhất)
→ Nhưng với lookahead, model biết A dẫn đến chi phí cao sau này
→ Nên chọn C (mặc dù xa) để minimize total cost
```

**Training lookahead**:
- Được train cùng với policy (actor-critic style)
- Target: actual return từ step tiếp theo đến hết episode
- Loss: MSE giữa prediction và actual return

**Ý nghĩa**:
- **Myopic avoidance**: Tránh greedy choices
- **Long-term planning**: Xem xét hậu quả dài hạn
- **Isolated customer handling**: 
  - Customer xa nhưng trong cluster → low lookahead → nên đi sớm
  - Customer gần nhưng isolated → high lookahead → nên đi sau hoặc tránh

---

### 8. Score Fusion - Kết hợp tín hiệu

**Chức năng**: Học cách kết hợp 3 nguồn tín hiệu thành final compatibility score.

**Kiến trúc**:
```
ScoreFusion (Advanced MLP)
├── Linear(3 → 64)
├── ReLU
└── Linear(64 → 1)
```

**Inputs** (3 sources):
1. `att_score`: $N \times 1 \times L_c$ - Attention compatibility (từ CrossEdgeFusion)
2. `owner_bias`: $N \times 1 \times L_c$ - Ownership log-probability (từ OwnershipHead)
3. `lookahead`: $N \times 1 \times L_c$ - Future cost estimate (từ LookaheadHead)

**Processing**:

```python
def _score_customers(veh_repr, cust_repr, edge_emb, owner_bias, lookahead):
    """
    Fuse multiple signals into final compatibility scores
    """
    # 1. Compute attention score
    att_score = cross_fusion(veh_repr, cust_repr, edge_emb)  # N × 1 × L_c
    
    # 2. Z-normalization (standardize each signal)
    def z_norm(s):
        mean = s.mean(dim=-1, keepdim=True)
        std = s.std(dim=-1, keepdim=True)
        return (s - mean) / (std + 1e-8)
    
    att_score_norm = z_norm(att_score)
    owner_bias_norm = z_norm(owner_bias)
    lookahead_norm = z_norm(lookahead)
    
    # 3. Stack signals
    combined = torch.stack([
        att_score_norm,
        owner_bias_norm,
        lookahead_norm
    ], dim=-1)  # N × 1 × L_c × 3
    
    # 4. Non-linear fusion
    compat = score_fusion(combined).squeeze(-1)  # N × 1 × L_c
    
    # 5. Optional tanh exploration
    if tanh_xplor is not None:
        compat = tanh_xplor * torch.tanh(compat)
    
    return compat
```

**Tại sao Z-normalization?**

```
Example WITHOUT z-norm:
att_score   = [10.0, 8.0, 6.0, 4.0, 2.0]
owner_bias  = [-2.0, -1.5, -1.0, -0.5, 0.0]
lookahead   = [100, 80, 60, 40, 20]

→ Lookahead dominates vì scale lớn hơn nhiều
→ Model không học được trọng số cân bằng

Example WITH z-norm:
att_score_norm   = [1.41, 0.71, 0.0, -0.71, -1.41]
owner_bias_norm  = [-1.41, -0.71, 0.0, 0.71, 1.41]
lookahead_norm   = [1.41, 0.71, 0.0, -0.71, -1.41]

→ Tất cả signals có scale giống nhau
→ MLP học được importance weights công bằng
```

**Ưu điểm của MLP fusion vs fixed weights**:

```python
# Fixed weights (old approach):
compat = w1 * att_score + w2 * owner_bias - w3 * lookahead

# MLP fusion (new approach):
compat = MLP([att_score, owner_bias, lookahead])

Advantages:
1. Non-linear interactions: MLP học được phức tạp hơn linear combination
   Ví dụ: "Nếu att_score cao VÀ owner_bias cao → boost nhiều hơn"
            "Nếu lookahead cao NHƯNG owner_bias rất cao → vẫn chọn"

2. Adaptive weights: Weights thay đổi theo context
   Ví dụ: Early episode → chú trọng att_score (explore)
          Late episode → chú trọng lookahead (exploit)

3. Learned từ data: Không cần hand-tune w1, w2, w3
```

**Tanh exploration**:

```python
if tanh_xplor is not None:
    compat = tanh_xplor * torch.tanh(compat)

Effect:
- tanh giới hạn compat vào [-1, 1]
- Nhân với tanh_xplor (default 10) → [-10, 10]
- Stabilize training: tránh exploding logits
- Control exploration: tanh_xplor cao → logits spread nhiều → explore nhiều
```

**Ví dụ đầy đủ**:

```
Scenario: 4 customers

Raw signals:
att_score   = [8.0, 6.0, 10.0, 4.0]   # Customer 2 có attention cao nhất
owner_bias  = [-1.0, -2.0, -0.5, -3.0] # Customer 2 có ownership cao nhất
lookahead   = [50, 30, 80, 20]        # Customer 3 có lookahead thấp nhất (tốt)

After z-norm:
att_score_norm   = [0.27, -0.54, 1.08, -0.81]
owner_bias_norm  = [0.27, -0.54, 1.08, -0.81]
lookahead_norm   = [0.27, -0.54, 1.08, -0.81]

MLP fusion:
combined = stack → N × 1 × 4 × 3
compat = MLP(combined) → [0.8, 0.3, 1.5, -0.2]

After tanh(compat) * 10:
final_compat = [6.7, 2.9, 9.1, -2.0]

→ Customer 2 có final score cao nhất
→ Kết hợp cả 3 yếu tố: attention, ownership, và lookahead
```

---

## Execution Flow - Luồng thực thi

### Khởi tạo (forward() function)

```python
def forward(dyna):
    """
    Main training loop
    
    Args:
        dyna: Dynamic environment
    
    Returns:
        actions: List of (veh_idx, cust_idx) tuples
        logps: List of log-probabilities
        rewards: List of immediate rewards
    """
    # 1. Reset environment
    dyna.reset()
    
    # 2. Initialize memory
    self._reset_memory(dyna)
    # memory: N × L_v × H, initialized to zeros
    
    # 3. Encode customers (once at beginning)
    self._encode_customers(dyna.nodes, dyna.cust_mask)
    # self.cust_repr: N × L_c × D, cached
    
    # 4. Main loop
    actions, logps, rewards = [], [], []
    
    while not dyna.done:
        # Check if new customers appeared (dynamic)
        if dyna.new_customers:
            self._encode_customers(dyna.nodes, dyna.cust_mask)
        
        # Make decision for current vehicle
        cust_idx, logp = self.step(dyna)
        
        # Record
        actions.append((dyna.cur_veh_idx, cust_idx))
        logps.append(logp)
        
        # Step environment
        reward = dyna.step(cust_idx)
        rewards.append(reward)
    
    return actions, logps, rewards
```

### Single Step (step() function)

```python
def step(dyna):
    """
    Make one decision for the current acting vehicle
    
    Args:
        dyna.vehicles: N × L_v × D_v
        dyna.nodes: N × L_c × D_c (customers)
        dyna.cur_veh_idx: N × 1 (which vehicle is acting)
        dyna.cur_veh_mask: N × 1 × L_c (feasibility mask)
        dyna.mask: N × L_v × L_c (full feasibility)
        
        self.cust_repr: N × L_c × D (cached from _encode_customers)
        self._veh_memory: N × L_v × H (maintained across steps)
    
    Returns:
        cust_idx: N × 1 (selected customer)
        logp: N × 1 (log-probability of selection)
    """
    
    # ===== STEP 1: Encode acting vehicle =====
    # Build vehicle representation with customer context
    veh_repr = self._repr_vehicle(
        dyna.vehicles, 
        dyna.cur_veh_idx, 
        dyna.mask
    )
    # veh_repr: N × 1 × D
    
    # Details:
    #   fleet_repr = fleet_encoder(vehicles, cust_repr, mask)
    #   veh_repr = fleet_repr.gather(1, cur_veh_idx)
    
    # ===== STEP 2: Build and encode edge features =====
    edge_feat = self._build_edge_features(
        dyna.vehicles,
        dyna.nodes,
        dyna.cur_veh_idx,
        dyna.cur_veh_mask
    )
    # edge_feat: N × 1 × L_c × 8
    # Features: distance, travel_time, arrival, wait, late, slack, feasible, cap_gap
    
    edge_emb = self.edge_encoder(edge_feat)
    # edge_emb: N × 1 × L_c × D
    
    # ===== STEP 3: Compute attention score =====
    att_score = self.cross_fusion(veh_repr, self.cust_repr, edge_emb)
    # att_score: N × 1 × L_c
    
    # ===== STEP 4: Compute ownership bias =====
    owner_logits = self.owner_head(self._veh_memory, self.cust_repr)
    # owner_logits: N × L_v × L_c
    
    owner_prob = F.softmax(owner_logits, dim=1)
    # owner_prob: N × L_v × L_c (softmax over vehicles)
    
    owner_bias = owner_prob.gather(
        1, 
        dyna.cur_veh_idx[:, :, None].expand(-1, -1, L_c)
    )
    # owner_bias: N × 1 × L_c (extract for acting vehicle)
    
    owner_bias = owner_bias.clamp(min=1e-9).log()
    # Convert to log-prob
    
    # ===== STEP 5: Compute lookahead =====
    lookahead = self.lookahead_head(veh_repr, self.cust_repr, edge_emb)
    # lookahead: N × 1 × L_c
    
    # ===== STEP 6: Fuse scores =====
    compat = self._score_customers(
        veh_repr,
        self.cust_repr,
        edge_emb,
        owner_bias,
        lookahead
    )
    # compat: N × 1 × L_c
    # Details:
    #   1. Z-normalize each signal
    #   2. Stack: [att_score, owner_bias, lookahead]
    #   3. MLP fusion
    #   4. Optional tanh exploration
    
    # ===== STEP 7: Apply mask and select =====
    logp = self._get_logp(compat, dyna.cur_veh_mask)
    # logp: N × L_c
    # Details:
    #   compat[mask] = -inf
    #   logp = log_softmax(compat, dim=-1)
    
    if self.greedy:
        cust_idx = logp.argmax(dim=1, keepdim=True)
    else:
        cust_idx = logp.exp().multinomial(1)
    # cust_idx: N × 1
    
    # ===== STEP 8: Update memory =====
    self._update_memory(
        dyna.cur_veh_idx,
        cust_idx,
        veh_repr,
        edge_emb
    )
    # self._veh_memory: N × L_v × H (updated)
    # Details:
    #   1. Gather selected customer: cust_sel = cust_repr.gather(1, cust_idx)
    #   2. Gather selected edge: edge_sel = edge_emb.gather(2, cust_idx)
    #   3. Update memory for acting vehicle only
    
    return cust_idx, logp.gather(1, cust_idx)
```

### Flowchart chi tiết

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                               │
├─────────────────────────────────────────────────────────────────────┤
│ dyna.reset()                                                         │
│ memory ← zeros(N × L_v × H)                                          │
│ _encode_customers(dyna.nodes, dyna.cust_mask)                        │
│   ├─ depot_emb ← depot_embedding(nodes[:, 0, :])                     │
│   ├─ cust_emb ← cust_embedding(nodes[:, 1:, :])                      │
│   ├─ h ← cat([depot_emb, cust_emb])                                  │
│   ├─ cust_enc ← GraphEncoder(h, mask, coords)                        │
│   │   └─ Multi-layer graph attention với RBF bias                    │
│   └─ cust_repr ← cust_project(cust_enc)  [CACHED]                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN LOOP: while not done                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
    ┌──────────────────────┐        ┌──────────────────────────┐
    │ if new_customers:    │        │ cur_veh_idx, cur_veh_mask│
    │ re-encode customers  │        │ (from environment)        │
    └──────────────────────┘        └──────────────────────────┘
                │                                 │
                └────────────────┬────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         STEP() FUNCTION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ PHASE 1: Vehicle Representation                               │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ fleet_repr ← FleetEncoder(vehicles, cust_repr, mask)         │  │
│  │   └─ Multi-layer cross-attention: vehicles attend to custs   │  │
│  │ veh_repr ← fleet_repr.gather(1, cur_veh_idx)                 │  │
│  │   └─ Extract acting vehicle: N × 1 × D                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                 │                                    │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ PHASE 2: Edge Features                                        │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ edge_feat ← _build_edge_features(...)                        │  │
│  │   ├─ distance ← cdist(veh_pos, cust_pos)                     │  │
│  │   ├─ travel_time ← distance / speed                          │  │
│  │   ├─ arrival ← veh_time + travel_time                        │  │
│  │   ├─ wait ← max(0, tw_start - arrival)                       │  │
│  │   ├─ late ← max(0, arrival - tw_end)                         │  │
│  │   ├─ slack ← tw_end - arrival                                │  │
│  │   ├─ feasible ← 1.0 or 0.0                                   │  │
│  │   └─ cap_gap ← veh_capa - cust_demand                        │  │
│  │ edge_emb ← EdgeFeatureEncoder(edge_feat)                     │  │
│  │   └─ MLP: Linear → ReLU → Linear → LayerNorm                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                 │                                    │
│            ┌────────────────────┼────────────────────┐              │
│            │                    │                    │              │
│            ▼                    ▼                    ▼              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ PHASE 3a:       │  │ PHASE 3b:       │  │ PHASE 3c:       │    │
│  │ Attention Score │  │ Ownership Bias  │  │ Lookahead       │    │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤    │
│  │ att_score ←     │  │ owner_logits ←  │  │ lookahead ←     │    │
│  │ CrossEdgeFusion │  │ OwnershipHead   │  │ LookaheadHead   │    │
│  │ (veh_repr,      │  │ (veh_memory,    │  │ (veh_repr,      │    │
│  │  cust_repr,     │  │  cust_repr)     │  │  cust_repr,     │    │
│  │  edge_emb)      │  │                 │  │  edge_emb)      │    │
│  │                 │  │ owner_prob ←    │  │                 │    │
│  │ Multi-head attn │  │ softmax(logits) │  │ MLP(concat(     │    │
│  │ + edge bias     │  │                 │  │   veh, cust,    │    │
│  │                 │  │ owner_bias ←    │  │   edge))        │    │
│  │ N × 1 × L_c     │  │ log(gather(...))│  │                 │    │
│  │                 │  │                 │  │ N × 1 × L_c     │    │
│  │                 │  │ N × 1 × L_c     │  │                 │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│            │                    │                    │              │
│            └────────────────────┼────────────────────┘              │
│                                 │                                    │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ PHASE 4: Score Fusion                                         │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ att_score_norm ← z_normalize(att_score)                      │  │
│  │ owner_bias_norm ← z_normalize(owner_bias)                    │  │
│  │ lookahead_norm ← z_normalize(lookahead)                      │  │
│  │                                                                │  │
│  │ combined ← stack([att_score_norm,                            │  │
│  │                   owner_bias_norm,                           │  │
│  │                   lookahead_norm], dim=-1)                   │  │
│  │   Shape: N × 1 × L_c × 3                                      │  │
│  │                                                                │  │
│  │ compat ← ScoreFusion(combined).squeeze(-1)                   │  │
│  │   └─ MLP: Linear(3→64) → ReLU → Linear(64→1)                 │  │
│  │                                                                │  │
│  │ if tanh_xplor:                                                │  │
│  │   compat ← tanh_xplor * tanh(compat)                         │  │
│  │   Shape: N × 1 × L_c                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                 │                                    │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ PHASE 5: Selection                                            │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ compat[cur_veh_mask] ← -inf  (mask infeasible)               │  │
│  │ logp ← log_softmax(compat, dim=-1)                           │  │
│  │   Shape: N × L_c                                              │  │
│  │                                                                │  │
│  │ if greedy:                                                    │  │
│  │   cust_idx ← argmax(logp)                                    │  │
│  │ else:                                                         │  │
│  │   cust_idx ← multinomial(exp(logp))                          │  │
│  │   Shape: N × 1                                                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                 │                                    │
│                                 ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ PHASE 6: Memory Update                                        │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ cust_sel ← cust_repr.gather(1, cust_idx)                     │  │
│  │ edge_sel ← edge_emb.gather(2, cust_idx)                      │  │
│  │                                                                │  │
│  │ context ← concat([veh_repr, cust_sel, edge_sel])             │  │
│  │ cur_h ← memory.gather(1, cur_veh_idx)                        │  │
│  │ next_h ← tanh(input_proj(context) + hidden_proj(cur_h))      │  │
│  │                                                                │  │
│  │ memory.scatter_(1, cur_veh_idx, next_h)                      │  │
│  │   └─ Only update acting vehicle's memory slot                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    Return (cust_idx, logp[cust_idx])
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ENVIRONMENT STEP                                │
├─────────────────────────────────────────────────────────────────────┤
│ reward ← dyna.step(cust_idx)                                         │
│   ├─ Update vehicle state (move to customer)                        │
│   ├─ Update time, capacity                                          │
│   ├─ Mark customer as served                                        │
│   ├─ Check if new customers arrived (dynamic)                       │
│   ├─ Select next acting vehicle                                     │
│   └─ Compute reward (negative travel time/distance/late penalty)    │
│                                                                      │
│ actions.append((cur_veh_idx, cust_idx))                             │
│ logps.append(logp)                                                   │
│ rewards.append(reward)                                               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                         Loop until done
                                 │
                                 ▼
                  Return (actions, logps, rewards)
```

---

## Hyperparameters & Configuration

### Model Architecture
```python
EdgeEnhencedLearner(
    cust_feat_size=7,       # (x, y, demand, tw_start, tw_end, ...)
    veh_state_size=4,       # (x, y, capacity, time)
    model_size=128,         # D: embedding dimension
    layer_count=2,          # Number of Transformer layers
    head_count=4,           # Number of attention heads
    ff_size=258,            # Feed-forward hidden size
    tanh_xplor=10,          # Exploration amplitude
    greedy=False,           # Sampling (False) or greedy (True)
    edge_feat_size=8,       # Edge feature dimension
    cust_k=20,              # k-NN for graph encoder
    memory_size=None,       # Memory size (default = model_size)
    lookahead_hidden=128,   # Lookahead MLP hidden size
    dropout=0.1             # Dropout rate
)
```

### Training Hyperparameters
```python
optimizer = Adam(lr=1e-4)
batch_size = 512
epochs = 500
baseline = "critic"  
```

---

## Key Innovations

### 1. Edge-Enhanced Representation
- **Problem**: Traditional methods treat vehicle-customer compatibility as pure attention
- **Solution**: Explicit edge features (time windows, capacity, distance) encoded separately
- **Impact**: Model understands constraints better → higher feasibility

### 2. Score Fusion with Normalization
- **Problem**: Different signals (attention, ownership, lookahead) có scales khác nhau
- **Solution**: Z-normalization + MLP fusion
- **Impact**: Balanced contribution, learned non-linear interactions

### 3. Coordination Memory
- **Problem**: Multiple vehicles conflict (choose same customers)
- **Solution**: Per-vehicle memory + ownership prediction
- **Impact**: Implicit coordination without communication

### 4. Lookahead Head
- **Problem**: Greedy selection leads to poor long-term performance
- **Solution**: Value network estimates future cost
- **Impact**: Better handling of isolated customers, reduced detours

### 5. Graph Encoder with RBF Bias
- **Problem**: Standard Transformer ignores spatial structure
- **Solution**: RBF distance encoding + k-NN sparsification
- **Impact**: Better spatial reasoning, lower complexity


---

## File Structure

```
mardam-master/
├── MODEL/
│   └── model/
│       └── marl_model.py          # Main model (EdgeEnhencedLearner)
├── layers/
│   ├── Mymodel_layers.py          # Custom layers (GraphEncoder, FleetEncoder, ...)
│   ├── _mha.py                    # Multi-head attention
│   ├── _transformer.py            # Standard Transformer layers
│   └── _loss.py                   # Loss functions
├── problems/
│   ├── _env_tw.py                 # VRPTW environment
│   ├── _env_dtw.py                # DVRPTW environment
│   └── _data_tw.py                # Data generation
├── script/
│   ├── train.py                   # Training script
│   ├── eval_learned_dyn.py        # Evaluation script
│   └── plot_routes.py             # Visualization
└── requirements.txt
```

---
