# COAST / VECTRA for Dynamic VRPTW

Repository này triển khai một policy học tăng cường cho bài toán **Dynamic Vehicle Routing Problem with Time Windows (DVRPTW)**. Tên nghiên cứu trong tài liệu/paper là **COAST**; tên implementation hiện tại trong mã nguồn vẫn là **VECTRA** (`MODEL/model/vectra.py`, class `VECTRA`).

Mục tiêu của mô hình là tách quyết định tuần tự thành các tín hiệu rõ ràng hơn:

- **Edge-aware compatibility**: xe hiện tại phù hợp với khách nào dưới ràng buộc khoảng cách, time window và capacity.
- **Coordination / ownership**: bộ nhớ riêng của từng xe tạo bias phân công mềm, giúp giảm chồng lấn giữa các xe.
- **Candidate-conditioned lookahead**: một scalar theo từng ứng viên khách hàng, học end-to-end để cung cấp tín hiệu dự đoán hệ quả quyết định.

Lưu ý quan trọng: trong code hiện tại, `LookaheadHead` và `OwnershipHead` **không có auxiliary supervised loss riêng**. Chúng được học gián tiếp qua policy gradient, critic baseline/PPO và fusion score cuối cùng.

## Thành Phần Chính

```
mardam-master/
├── MODEL/
│   ├── model/vectra.py        # Mô hình đề xuất COAST/VECTRA
│   ├── train.py               # Huấn luyện REINFORCE/actor-critic baseline
│   ├── train_PPO.py           # Huấn luyện PPO thay thế
│   ├── infer.py               # Suy luận COAST/VECTRA
│   └── infer_mardam.py        # Suy luận baseline MARDAM/AttentionLearner cũ
├── layers/
│   ├── Mymodel_layers.py      # GraphEncoder, FleetEncoder, EdgeFeatureEncoder, ...
│   ├── _loss.py               # reinforce_loss
│   ├── _mha.py                # Multi-head attention
│   └── _transformer.py        # Transformer cũ dùng cho AttentionLearner
├── problems/
│   ├── _data_dtw.py           # DVRPTW_Dataset, CSV loader, synthetic generator
│   ├── _env_dtw.py            # DVRPTW_Environment
│   ├── _data_tw.py            # VRPTW_Dataset
│   └── _env_tw.py             # VRPTW_Environment
├── baselines/                 # none, nearest-neighbour, rollout, critic
├── utils/
│   ├── _args.py               # CLI args và ablation profiles
│   └── _chkpt.py              # checkpoint/resume/model-weight loading
├── script/
│   ├── train_vectra_main.sh   # Lệnh train COAST/VECTRA mặc định
│   ├── train_mardam.py        # Train baseline AttentionLearner cũ
│   ├── train_mardam.sh
│   ├── run_ablation_study.sh
│   ├── eval_unified.py
│   └── infer_all_datasets.py
└── data/
    ├── vectra/                # checkpoint COAST/VECTRA có sẵn
    ├── mardam/                # checkpoint baseline có sẵn
    └── _Ablation/             # checkpoint ablation có sẵn
```

## Mô Hình Đề Xuất

Entrypoint của mô hình là:

```python
from MODEL.model import VECTRA
```

Constructor chính:

```python
VECTRA(
    cust_feat_size,
    veh_state_size,
    model_size=128,
    layer_count=2,
    head_count=4,
    ff_size=256,
    tanh_xplor=10,
    greedy=False,
    edge_feat_size=8,
    memory_size=128,
    lookahead_hidden=128,
    dropout=0.1,
    adaptive_depth=False,
    adaptive_min_layers=1,
    adaptive_easy_ratio=0.6,
    latent_bottleneck=False,
    latent_tokens=32,
    latent_min_nodes=64,
    use_edge_features=True,
    use_memory=True,
    use_ownership=True,
    use_lookahead=True,
    fusion_mode="mlp",
    linear_fusion_weights=(1.0, 1.0, 1.0),
)
```

### 1. Customer Encoding

`VECTRA._encode_customers()` tạo embedding riêng cho depot và customer, sau đó đưa qua `GraphEncoder`.

`GraphEncoder` trong `layers/Mymodel_layers.py` dùng:

- multi-head self-attention trên các node khách hàng;
- RBF distance bias từ ma trận khoảng cách Euclidean;
- tùy chọn `adaptive_depth` để dùng ít layer hơn với instance dễ;
- tùy chọn `latent_bottleneck` để encode một tập token rút gọn rồi gán representation về toàn bộ node theo nearest token.

Input node feature:

- VRP: $(x, y, \text{demand})$.
- VRPTW: $(x, y, \text{demand}, \text{open}, \text{close}, \text{service\_time})$.
- DVRPTW: $(x, y, \text{demand}, \text{open}, \text{close}, \text{service\_time}, \text{appearance\_time})$.

**Công thức:**

Gọi $\mathbf{x}_j \in \mathbb{R}^{D_c}$ là feature thô của node $j$ (node $0$ là depot, các node $j \ge 1$ là khách hàng).

**Embedding ban đầu:**

$$\mathbf{h}_j^{(0)} = \begin{cases} \mathbf{W}_{\text{depot}} \mathbf{x}_j, & j = 0 \\ \mathbf{W}_{\text{cust}} \mathbf{x}_j, & j \ge 1 \end{cases}$$

**Multi-head Self-Attention với RBF distance bias (tại layer $l$):**

$$\mathbf{Q}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_Q^{(l)}, \quad \mathbf{K}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_K^{(l)}, \quad \mathbf{V}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_V^{(l)}$$

$$\alpha_{ij}^{(l,h)} = \text{softmax}_j \left( \frac{\mathbf{q}_i^{(l,h)} \cdot \mathbf{k}_j^{(l,h)}}{\sqrt{d_k}} + \text{RBF}(d_{ij}) \cdot \mathbf{w}_{\text{edge}}^{(l,h)} \right)$$

$$\hat{\mathbf{h}}_i^{(l)} = \mathbf{h}_i^{(l)} + \mathbf{W}_{\text{out}}^{(l)} \left[ \bigoplus_{h=1}^{H} \sum_j \alpha_{ij}^{(l,h)} \mathbf{v}_j^{(l,h)} \right]$$

$$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\left( \hat{\mathbf{h}}_i^{(l)} + \text{FFN}(\hat{\mathbf{h}}_i^{(l)}) \right), \quad \text{FFN}(\mathbf{z}) = \mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{z})$$

Trong đó $d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|_2$ là khoảng cách Euclidean, được chuẩn hóa về $[0, 1]$ trong batch.

**RBF Kernel (16 bins):**

$$\text{RBF}_k(d) = \exp \left( -\frac{(d - \mu_k)^2}{2\sigma^2} \right), \quad \mu_k \in \left\{0, \tfrac{1}{15}, \dots, 1\right\}, \quad \sigma = \tfrac{1}{15}$$

$$\text{RBF}(d_{ij}) \in \mathbb{R}^{16} \xrightarrow{\text{MLP}} \mathbb{R}^{H}$$

**Adaptive Depth:** số layer thực tế được chọn dựa trên tỉ lệ node visible:

$$L_{\text{used}} = \begin{cases} L_{\text{min}}, & \rho \ge \rho_{\text{easy}} \\ L_{\text{min}} + \lfloor (L - L_{\text{min}}) \cdot \frac{\rho_{\text{easy}} - \rho}{\rho_{\text{easy}}} \rfloor, & \rho < \rho_{\text{easy}} \end{cases}$$

với $\rho = 1 - \frac{|\text{masked nodes}|}{L_c}$, $\rho_{\text{easy}}$ = `adaptive_easy_ratio`.

**Latent Bottleneck:** chọn $T \ll L_c$ token đại diện (bao gồm depot) theo lưới đều, encode qua GraphEncoder, rồi gán representation về toàn bộ node theo nearest neighbor:

$$\mathbf{h}_j = \mathbf{h}_{\text{token}\left(\arg\min_k d_{jk}\right)}^{\text{enc}}$$

**Final Projection:**

$$\mathbf{c}_j = \text{Dropout}\left( \mathbf{W}_c \mathbf{h}_j^{(L)} \right)$$

với $\mathbf{c}_j \in \mathbb{R}^{D}$ là customer representation cuối cùng (`cust_repr`).

### 2. Acting Vehicle Encoding

Ở code hiện tại, mỗi bước chỉ encode **xe đang hành động**. `_repr_vehicle()` gather state của `cur_veh_idx`, rồi gọi `FleetEncoder` để xe hiện tại cross-attend tới `cust_repr`.

Vehicle state có 4 chiều:

$$\mathbf{s}_i = (x_i, y_i, q_i^{\text{rem}}, t_i^{\text{cur}}) \in \mathbb{R}^{4}$$

Phối hợp giữa nhiều xe không đến từ self-attention trực tiếp giữa các vehicle trong bước này; nó đến từ `CoordinationMemory` và `OwnershipHead`.

**Công thức:**

Tại mỗi bước, chỉ xe đang hành động $i^*$ được encode. Gọi $\mathbf{s}_{i^*} \in \mathbb{R}^{1 \times 4}$ là state của xe đó.

**Input Projection:**

$$\mathbf{v}_{i^*}^{(0)} = \mathbf{W}_{\text{in}} \mathbf{s}_{i^*}$$

**Cross-Attention với Customer Representation (tại layer $l$):**

$$\mathbf{v}_{i^*}^{(l+1)} = \text{LayerNorm} \left( \mathbf{v}_{i^*}^{(l)} + \text{MHA}\left(\mathbf{v}_{i^*}^{(l)}, \mathbf{C}, \mathbf{C}, \mathbf{M}_{i^*}\right) \right)$$

$$\mathbf{v}_{i^*}^{(l+1)} = \text{LayerNorm} \left( \mathbf{v}_{i^*}^{(l+1)} + \text{FFN}(\mathbf{v}_{i^*}^{(l+1)}) \right)$$

trong đó $\mathbf{C} = [\mathbf{c}_0, \mathbf{c}_1, \dots, \mathbf{c}_{L_c-1}]^\top \in \mathbb{R}^{L_c \times D}$ là ma trận customer representation, và $\mathbf{M}_{i^*} \in \{0,1\}^{1 \times L_c}$ là mask feasibility của xe $i^*$.

**Final Projection:**

$$\tilde{\mathbf{v}}_{i^*} = \mathbf{W}_v \mathbf{v}_{i^*}^{(L)} \in \mathbb{R}^{1 \times D}$$

Đây chính là `veh_repr` — representation của xe đang hành động.

### 3. Edge Features

`_build_edge_features()` tạo đặc trưng cho từng cặp `(acting vehicle, customer)` với 8 chiều:

| Feature | Ký hiệu | Công thức |
|---|---|---|
| `distance` | $d_{ij}$ | $\|\mathbf{p}_i - \mathbf{p}_j\|_2$ |
| `travel_time` | $\tau_{ij}$ | $d_{ij} / v_{\text{speed}}$ |
| `arrival` | $a_{ij}$ | $t_i^{\text{cur}} + \tau_{ij}$ |
| `wait` | $w_{ij}$ | $\max(o_j - a_{ij}, 0)$ |
| `late` | $l_{ij}$ | $\max(a_{ij} - c_j, 0)$ |
| `slack` | $s_{ij}$ | $c_j - a_{ij}$ |
| `feasible` | $f_{ij}$ | $\mathbb{1}[j \in \mathcal{F}_i]$ |
| `cap_gap` | $g_{ij}$ | $q_i^{\text{rem}} - \text{demand}_j$ |

Trong đó $o_j, c_j$ là open/close time window, $\mathcal{F}_i$ là tập khách hàng khả thi cho xe $i$.

**Edge Feature Encoder (MLP):**

$$\mathbf{e}_{ij} = [d_{ij}, \tau_{ij}, a_{ij}, w_{ij}, l_{ij}, s_{ij}, f_{ij}, g_{ij}] \in \mathbb{R}^{8}$$

$$\mathbf{e}_{ij}^{\text{emb}} = \text{LayerNorm}\left( \mathbf{W}_{e2} \; \text{ReLU}\left( \mathbf{W}_{e1} \mathbf{e}_{ij} \right) \right) \in \mathbb{R}^{D}$$

### 4. Cross-Edge Fusion

`CrossEdgeFusion` tính attention score giữa vehicle representation và customer representation, rồi cộng thêm edge bias từ `edge_emb`.

Kết quả:

```text
att_score: N x 1 x L_c
```

Đây là compatibility cơ sở, trước khi thêm ownership và lookahead.

**Công thức:**

Gọi $\tilde{\mathbf{v}} \in \mathbb{R}^{1 \times D}$ là vehicle representation, $\mathbf{C} \in \mathbb{R}^{L_c \times D}$ là customer representation, và $\mathbf{E}^{\text{emb}} \in \mathbb{R}^{1 \times L_c \times D}$ là edge embedding.

**Multi-head Attention với Edge Bias:**

$$\mathbf{q}^{(h)} = \tilde{\mathbf{v}} \mathbf{W}_{Q, \text{fuse}}^{(h)}, \quad \mathbf{k}_j^{(h)} = \mathbf{c}_j \mathbf{W}_{K, \text{fuse}}^{(h)}$$

$$b_{j}^{(h)} = \mathbf{e}_{j}^{\text{emb}} \cdot \mathbf{w}_{\text{bias}}^{(h)}$$

$$\alpha_j^{(h)} = \text{softmax}_j \left( \frac{\mathbf{q}^{(h)} \cdot \mathbf{k}_j^{(h)}}{\sqrt{d_h}} + b_j^{(h)} \right)$$

**Pooling qua các head (mean):**

$$s_{ij}^{\text{att}} = \frac{1}{H} \sum_{h=1}^{H} \alpha_j^{(h)} \in \mathbb{R}^{1 \times L_c}$$

### 5. Coordination Memory Và Ownership

`CoordinationMemory` duy trì hidden state riêng cho từng xe:

```text
_veh_memory: N x L_v x H
```

Memory khởi tạo bằng 0 khi reset rollout. Sau mỗi quyết định, chỉ slot của xe vừa hành động được cập nhật bằng:

```text
tanh(input_proj([veh_repr, selected_customer_repr, selected_edge_emb]) + hidden_proj(old_memory))
```

`OwnershipHead` dùng toàn bộ `_veh_memory` và `cust_repr` để tạo logits:

```text
owner_logits: N x L_v x L_c
```

Sau đó softmax theo chiều vehicle, gather xác suất ứng với acting vehicle, rồi lấy log để thành:

```text
owner_bias: N x 1 x L_c
```

Nếu tắt `--disable-memory` hoặc `--disable-ownership`, ownership bias được thay bằng tensor 0.

**Công thức:**

**Memory khởi tạo:**

$$\mathbf{M}^{(0)} = \mathbf{0} \in \mathbb{R}^{N \times L_v \times H}$$

**Memory update (chỉ cho xe $i^*$ vừa hành động, chọn khách $j^*$):**

$$\mathbf{m}_{i^*}^{(t+1)} = \tanh \left( \mathbf{W}_{\text{in}} \left[ \tilde{\mathbf{v}}_{i^*}, \mathbf{c}_{j^*}, \mathbf{e}_{i^* j^*}^{\text{emb}} \right] + \mathbf{W}_{\text{hid}} \mathbf{m}_{i^*}^{(t)} \right)$$

$$\mathbf{m}_{k}^{(t+1)} = \mathbf{m}_{k}^{(t)} \quad \forall k \neq i^*$$

trong đó $[\cdot, \cdot, \cdot]$ là concatenation, $\mathbf{W}_{\text{in}}$ là `LazyLinear` (input dim = $2D + D$, output dim = $H$), $\mathbf{W}_{\text{hid}} \in \mathbb{R}^{H \times H}$.

**Ownership Logits:**

$$\mathbf{V}^{\text{mem}} = \mathbf{W}_{\text{veh}} \mathbf{M}^{(t)} \in \mathbb{R}^{N \times L_v \times D}, \quad \mathbf{C}^{\text{proj}} = \mathbf{W}_{\text{cust}} \mathbf{C} \in \mathbb{R}^{N \times L_c \times D}$$

$$o_{ik} = \frac{\mathbf{v}_i^{\text{mem}} \cdot \mathbf{c}_k^{\text{proj}}}{\sqrt{D}} \in \mathbb{R}^{N \times L_v \times L_c}$$

**Ownership Bias (cho xe đang hành động $i^*$):**

$$p_{ik}^{\text{own}} = \frac{\exp(o_{ik})}{\sum_{i'=1}^{L_v} \exp(o_{i'k})}$$

$$\text{owner\_bias}_{k} = \log \left( p_{i^* k}^{\text{own}} \right) \in \mathbb{R}^{N \times 1 \times L_c}$$

### 6. Candidate-Conditioned Lookahead

`LookaheadHead` nhận `[veh_repr, cust_repr_j, edge_emb_j]` cho từng candidate và xuất scalar:

```text
lookahead: N x 1 x L_c
```

Trong implementation hiện tại, scalar này là **tín hiệu học được theo từng candidate**, không phải value/cost đã được calibrate bằng target MSE riêng. Fusion MLP học cách dùng tín hiệu này cùng attention và ownership để tăng/giảm score cuối.

Nếu tắt `--disable-lookahead`, lookahead được thay bằng tensor 0.

**Công thức:**

Với mỗi candidate $j$, lookahead head nhận concatenation của 3 nguồn thông tin:

$$\mathbf{f}_{ij} = [\tilde{\mathbf{v}}_i, \mathbf{c}_j, \mathbf{e}_{ij}^{\text{emb}}] \in \mathbb{R}^{3D}$$

$$\ell_{ij} = \mathbf{W}_{\ell 2} \; \text{ReLU}\left( \mathbf{W}_{\ell 1} \mathbf{f}_{ij} \right) \in \mathbb{R}$$

với $\mathbf{W}_{\ell 1} \in \mathbb{R}^{H_\ell \times 3D}$, $\mathbf{W}_{\ell 2} \in \mathbb{R}^{1 \times H_\ell}$, $H_\ell$ = `lookahead_hidden` (mặc định 128).

### 7. Score Fusion Và Chọn Hành Động

Mỗi bước, mô hình tạo ba nguồn score:

```text
att_score, owner_bias, lookahead
```

`_score_customers()` chuẩn hóa từng nguồn bằng mask-aware z-normalization trên các candidate hợp lệ. Sau đó:

- `fusion_mode="mlp"`: `Linear(3 -> 64) -> ReLU -> Linear(64 -> 1)`.
- `fusion_mode="linear"`: cộng tuyến tính theo `linear_fusion_weights`.

Nếu `tanh_xplor` khác `None`, logits được chặn bằng:

```text
compat = tanh_xplor * tanh(compat)
```

`_get_logp()` mask các candidate không hợp lệ bằng `-inf`, softmax theo customer, rồi:

- `greedy=True`: chọn `argmax`;
- `greedy=False`: sample bằng `multinomial`.

Nếu một batch row bị mask toàn bộ, code mở lại depot index `0` để tránh NaN.

**Công thức:**

Gọi $\mathcal{F}_i = \{j : \text{khách } j \text{ khả thi cho xe } i\}$ là tập candidate hợp lệ.

**Mask-aware Z-Normalization (cho từng nguồn score):**

$$\mu_i^{(\cdot)} = \frac{1}{|\mathcal{F}_i|} \sum_{j \in \mathcal{F}_i} s_{ij}^{(\cdot)}, \quad (\sigma_i^{(\cdot)})^2 = \frac{1}{|\mathcal{F}_i|} \sum_{j \in \mathcal{F}_i} \left( s_{ij}^{(\cdot)} - \mu_i^{(\cdot)} \right)^2$$

$$\tilde{s}_{ij}^{(\cdot)} = \frac{s_{ij}^{(\cdot)} - \mu_i^{(\cdot)}}{\sqrt{(\sigma_i^{(\cdot)})^2 + \epsilon}} \cdot \mathbb{1}[j \in \mathcal{F}_i], \quad \epsilon = 10^{-8}$$

với $(\cdot) \in \{\text{att}, \text{owner}, \text{look}\}$.

**Score Fusion:**

$$\text{MLP fusion:} \quad \text{compat}_{ij} = \mathbf{W}_{f2} \; \text{ReLU}\left( \mathbf{W}_{f1} [\tilde{s}_{ij}^{\text{att}}, \tilde{s}_{ij}^{\text{owner}}, \tilde{s}_{ij}^{\text{look}}] \right)$$

$$\text{Linear fusion:} \quad \text{compat}_{ij} = w_1 \tilde{s}_{ij}^{\text{att}} + w_2 \tilde{s}_{ij}^{\text{owner}} + w_3 \tilde{s}_{ij}^{\text{look}}$$

trong đó $\mathbf{W}_{f1} \in \mathbb{R}^{64 \times 3}$, $\mathbf{W}_{f2} \in \mathbb{R}^{1 \times 64}$, và $(w_1, w_2, w_3)$ = `linear_fusion_weights`.

**Tanh Exploration:**

$$\text{compat}_{ij} = \beta \cdot \tanh(\text{compat}_{ij}), \quad \beta = \text{tanh\_xplor}$$

**Action Selection:**

$$\log p(j | i) = \log \left( \frac{\exp(\text{compat}_{ij}) \cdot \mathbb{1}[j \in \mathcal{F}_i]}{\sum_{k \in \mathcal{F}_i} \exp(\text{compat}_{ik})} \right)$$

$$j^* = \begin{cases} \arg\max_j \log p(j|i), & \text{greedy} = \text{True} \\ \text{Categorical}(\exp(\log p(\cdot|i))), & \text{greedy} = \text{False} \end{cases}$$

## Kiến Trúc Tổng Quan

```
 ╔══════════════════════════════════════════════════════════════════╗
 ║                         ĐẦU VÀO                                ║
 ║  customers (x,y,demand,open,close,svc)  │  vehicles (x,y,capa,t) ║
 ║              cur_veh_idx i*                                     ║
 ╚══════════════════════╤══════════════════════╤═══════════════════╝
                        │                      │
          ┌─────────────┘                      └─────────────┐
          ▼                                                  ▼
 ╔══════════════════════════╗              ╔══════════════════════════╗
 ║ 1. CUSTOMER ENCODING     ║              ║ 2. EDGE FEATURES         ║
 ║                          ║              ║                          ║
 ║  ┌────────────────────┐  ║   customers  ║  ┌────────────────────┐  ║
 ║  │   GraphEncoder     │  ║ ───────────→ ║  │ _build_edge_feat() │  ║
 ║  │ Self-Attn + RBF    │  ║  vehicles    ║  │ 8 đặc trưng        │  ║
 ║  │ + optional depth   │  ║  cur_veh_idx ║  │ → EdgeFeatureEnc   │  ║
 ║  └────────┬───────────┘  ║              ║  └──────────┬─────────┘  ║
 ║           │ cust_repr    ║              ║             │ edge_emb   ║
 ╚═══════════╪══════════════╝              ╚═════════════╪════════════╝
             │                                           │
             │         ┌─────────────────────────────────┘
             │         │
             │         │   cur_veh_idx
             │         │   (gather vehicle state)
             │         │        │
             ├─────────┼────────┤
             ▼         │        ▼
 ╔══════════════════════════════════════╗
 ║ 3. VEHICLE ENCODING                  ║
 ║                                      ║
 ║  ┌────────────────────────────────┐  ║
 ║  │        FleetEncoder            │  ║
 ║  │  Cross-Attention:              │  ║
 ║  │    veh_state → Q               │  ║
 ║  │    cust_repr → K, V            │  ║
 ║  └──────────────┬─────────────────┘  ║
 ║                 │ veh_repr           ║
 ╚═════════════════╪════════════════════╝
                   │
                   │   veh_repr (N×1×D)
                   │
     ┌─────────────┼─────────────┬──────────────────┐
     │             │             │                  │
     │   cust_repr │             │ edge_emb         │
     │   (N×Lc×D)  │             │ (N×1×Lc×D)       │
     │             │             │                  │
     ▼             ▼             ▼                  ▼
 ╔══════════════╗ ╔══════════════╗ ╔══════════════════════════╗
 ║ 4a. CROSS-   ║ ║ 4b. LOOK-   ║ ║ 4c. OWNERSHIP HEAD       ║
 ║   EDGE FUSION║ ║   AHEAD     ║ ║                          ║
 ║              ║ ║   HEAD      ║ ║  ┌────────────────────┐  ║
 ║  Q·Kᵀ        ║ ║             ║ ║  │  veh_mem · custᵀ  │  ║
 ║  ──── + edge ║ ║ MLP(        ║ ║  │  ─────────────    │  ║
 ║   √dₖ        ║ ║  [v,c,e] )  ║ ║  │       √D          │  ║
 ║              ║ ║   → scalar  ║ ║  └────────┬───────────┘  ║
 ╚══════╤═══════╝ ╚══════╤═══════╝ ║           │ owner_bias  ║
        │ s_att           │ s_look  ╚═══════════╪═════════════╝
        │                 │                    │
        └─────────────────┼────────────────────┘
                          │
                          ▼
 ╔══════════════════════════════════════════════════════════════╗
 ║ 5. SCORE FUSION                                              ║
 ║                                                              ║
 ║   ┌──────────────────────────────────────────────────────┐   ║
 ║   │  Z-Normalization (mask-aware, per source):           │   ║
 ║   │    s̃ = (s − μ_F) / σ_F    (chỉ trên candidate hợp lệ) │   ║
 ║   │                                                      │   ║
 ║   │  MLP fusion:  Linear(3→64) → ReLU → Linear(64→1)    │   ║
 ║   │  Linear fusion: w₁·s_att + w₂·s_owner + w₃·s_look  │   ║
 ║   └──────────────────────────┬───────────────────────────┘   ║
 ║                              │ compat                        ║
 ╚══════════════════════════════╪═══════════════════════════════╝
                                │
                                ▼
 ╔══════════════════════════════════════════════════════════════╗
 ║ 6. ACTION SELECTION                                          ║
 ║                                                              ║
 ║   mask → softmax →  sample (train) / greedy (infer)          ║
 ║                                                              ║
 ║   log p(j|i) = log softmax(compat_ij · 𝟙[j ∈ F_i])          ║
 ╚══════════════════════════════╤═══════════════════════════════╝
                                │
                                │  j* = selected customer
                                ▼
 ╔══════════════════════════════════════════════════════════════╗
 ║ 7. HÀNH ĐỘNG  &  MEMORY UPDATE                              ║
 ║                                                              ║
 ║   dyna.step(j*)           ┌──────────────────────────────┐   ║
 ║                           │   CoordinationMemory         │   ║
 ║   [veh_repr, cust_j*,     │                              │   ║
 ║    edge_emb_j*]  ───────→ │ m(t+1)=tanh(Wᵢ·[v,c,e]      │   ║
 ║                           │          + Wₕ·m(t))          │   ║
 ║                           │ N × Lv × H                   │   ║
 ║                           └──────────────┬───────────────┘   ║
 ╚══════════════════════════════════════════╪════════════════════╝
                                           │
                                           │ veh_memory
                                           │ (quay lại step 4c)
                                           │
                        ┌──────────────────┘
                        │
                        ▼
                  (bước tiếp theo)
```

**Luồng dữ liệu chính:**

1. **Customer Encoding:** Các node khách hàng (gồm depot) được embed → `GraphEncoder` (self-attention + RBF distance bias) → `cust_repr`.
2. **Edge Features:** Cặp (xe, khách) → 8 đặc trưng → MLP → `edge_emb`.
3. **Vehicle Encoding:** State xe hiện tại (gathered by `cur_veh_idx`) + `cust_repr` → `FleetEncoder` (cross-attention) → `veh_repr`.
4. **Ba tín hiệu song song (đầu vào: `veh_repr` + `cust_repr` + `edge_emb`):**
   - **CrossEdgeFusion:** scaled dot-product attention + edge bias → $s^{\text{att}}$.
   - **LookaheadHead:** MLP($[\tilde{\mathbf{v}}, \mathbf{c}_j, \mathbf{e}_{j}^{\text{emb}}]$) → $s^{\text{look}}$.
   - **OwnershipHead:** $\mathbf{M} \cdot \mathbf{C}^\top / \sqrt{D}$ → softmax theo xe → log → $s^{\text{owner}}$ (chỉ dùng `cust_repr`, không dùng `edge_emb`).
5. **Score Fusion:** Z-normalization từng nguồn → MLP(3→64→1) hoặc linear weighted sum → `compat`.
6. **Action Selection:** `compat` → mask → softmax → sample/greedy → $j^*$.
7. **Memory Update:** `[veh_repr, cust_repr_{j^*}, edge_emb_{j^*}]` → $\tanh$ update → `veh_memory` mới, quay lại step 4c cho bước sau.

## Luồng Forward

```text
dyna.reset()
reset per-vehicle memory

while not dyna.done:
    if dyna.new_customers:
        encode customers again

    encode acting vehicle
    build edge features
    compute edge-aware attention score
    compute ownership bias from memory
    compute lookahead signal
    fuse scores
    sample/greedy-select next customer
    update acting vehicle memory
    dyna.step(customer)
```

Với DVRPTW, `DVRPTW_Environment` ẩn các khách có `appearance_time > current_time`. Khi khách mới được reveal, `dyna.new_customers=True`, làm customer representation được encode lại.

## Huấn Luyện

Script chính cho mô hình đề xuất:

```bash
PYTHONPATH=. python MODEL/train.py \
  --problem-type dvrptw \
  --customers-count 50 \
  --vehicles-count 3 \
  --epoch-count 500 \
  --iter-count 1000 \
  --batch-size 512 \
  --test-batch-size 10240 \
  --learning-rate 1e-4 \
  --model-size 128 \
  --layer-count 2 \
  --head-count 4 \
  --ff-size 256 \
  --edge-feat-size 8 \
  --memory-size 128 \
  --lookahead-hidden 128 \
  --baseline-type critic \
  --critic-rate 1e-3 \
  --max-grad-norm 2 \
  --amp \
  --output-dir output/vectra_run
```

Hoặc dùng wrapper:

```bash
./script/train_vectra_main.sh
```

`MODEL/train.py` làm các bước sau:

1. Sinh dữ liệu train online bằng `Dataset.generate(iter_count * batch_size, ...)`.
2. Normalize train/test dataset.
3. Tạo environment tương ứng với `--problem-type`.
4. Khởi tạo `VECTRA`.
5. Bọc policy bằng baseline: `none`, `nearnb`, `rollout` hoặc `critic`.
6. Tối ưu bằng Adam, tùy chọn LR decay, AMP và gradient clipping.
7. Tính loss bằng `reinforce_loss()`.
8. Lưu `args.json`, `train_statistics.csv`, `chkpt_best.pyth` và checkpoint định kỳ `chkpt_ep*.pyth`.

Với `baseline-type=critic`, critic là một MLP nhỏ trong `baselines/_critic.py`. Critic học bằng Smooth L1 loss trên return, còn policy học bằng REINFORCE advantage. `entropy_coef` có thể dùng để giữ exploration.

### PPO

PPO nằm trong:

```bash
PYTHONPATH=. python MODEL/train_PPO.py [các tham số giống train.py]
```

`train_PPO.py` collect rollout bằng policy hiện tại, tính GAE/return, rồi update bằng clipped PPO objective. Script này ép `baseline_type="critic"` và dùng cùng class `VECTRA`.

## Ablation Profiles

Các profile được định nghĩa trong `utils/_args.py`:

| Profile | Edge | Memory | Ownership | Lookahead | Fusion |
|---|---|---|---|---|---|
| `vectra` | on | on | on | on | MLP |
| `b0` / `a0` | on | off | off | off | MLP |
| `b1` / `a1` | on | on | off | off | MLP |
| `b3` / `a3` | on | off | off | on | MLP |
| `b5` / `a9` | on | on | on | on | linear |
| `edgeoff` / `a4` | off | on | on | on | MLP |
| `no_ownership` | on | on | off | on | MLP |
| `no_lookahead` | on | on | on | off | MLP |

Ví dụ:

```bash
PYTHONPATH=. python MODEL/train.py \
  --problem-type dvrptw \
  --customers-count 50 \
  --vehicles-count 3 \
  --ablation-profile b3 \
  --baseline-type critic \
  --output-dir output/ablation_b3
```

Lưu ý: parser hiện tại nhận `--ablation-profile vectra` cho full model, không nhận `coast`. Nếu dùng `script/run_ablation_study.sh`, nên truyền rõ:

```bash
ABLATION_PROFILE=vectra ./script/run_ablation_study.sh
```

## Suy Luận

Suy luận mô hình đề xuất:

```bash
PYTHONPATH=. python MODEL/infer.py \
  --problem-type dvrptw \
  --model-weight data/vectra/chkpt_best.pyth \
  --config-file data/vectra/args.json \
  --greedy \
  --max-print-instances 3 \
  --save-json output/infer_vectra.json
```

Để lưu diagnostics theo từng bước quyết định:

```bash
PYTHONPATH=. python MODEL/infer.py \
  --problem-type dvrptw \
  --model-weight data/vectra/chkpt_best.pyth \
  --config-file data/vectra/args.json \
  --greedy \
  --save-json output/infer_vectra_with_steps.json \
  --save-step-diagnostics \
  --step-diagnostics-limit 1
```

Có thể suy luận từ CSV DVRPTW:

```bash
PYTHONPATH=. python MODEL/infer.py \
  --problem-type dvrptw \
  --model-weight data/vectra/chkpt_best.pyth \
  --config-file data/vectra/args.json \
  --data-csv data/datasets/100/h100c101.csv \
  --vehicles-count 3 \
  --veh-capa 200 \
  --veh-speed 1 \
  --greedy \
  --save-json output/h100c101_vectra.json
```

CSV cần các cột:

```text
x,y,demand,open,close,servicetime
```

Cột `time` là optional và được dùng làm `appearance_time`; nếu thiếu thì mặc định bằng 0.

`MODEL/infer.py` có thể verify route bằng replay, kiểm tra skipped customers, duplicate customers, time-window violations, appearance violations và lưu các diagnostics này vào JSON.

## Baseline MARDAM / AttentionLearner Cũ

Baseline cũ dùng class `AttentionLearner` trong `_learner.py`. Đây không phải mô hình đề xuất hiện tại.

Các entrypoint liên quan:

```text
script/train_mardam.py
script/train_mardam.sh
MODEL/infer_mardam.py
```

Khác biệt chính:

- `AttentionLearner` dùng TransformerEncoder chuẩn và attention score trực tiếp `veh_repr @ cust_repr`.
- Không có edge feature encoder.
- Không có coordination memory.
- Không có ownership head.
- Không có candidate-conditioned lookahead.
- Không có score fusion nhiều tín hiệu.

## Đánh Giá Và Batch Inference

Đánh giá các checkpoint COAST/ablation theo thư mục seed:

```bash
PYTHONPATH=. python script/eval_unified.py \
  --test-data data/dvrptw_n50m3_10240.pyth \
  --models-dir output/ablation \
  --output output/eval_results/in_dist.json \
  --seeds 42,123,456,789,1024
```

Chạy inference hàng loạt trên các CSV:

```bash
PYTHONPATH=. python script/infer_all_datasets.py \
  --datasets-root data/datasets \
  --config-file data/vectra/args.json \
  --model-weight data/vectra/chkpt_best.pyth \
  --vehicles-count 3 \
  --veh-capa 200 \
  --veh-speed 1 \
  --output-dir output/batch_infer_vectra \
  --greedy
```

Sinh bộ artefact paper-ready từ raw summaries:

```bash
PYTHONPATH=. python script/build_experimental_report.py \
  --discover-nested \
  --results-root "Experimental result-20260526T064624Z-3-001/Experimental result/dynamic_benchmark"
```

Output chuẩn gồm `manifest.json`, `master_summary.csv`, `cell_summary.csv`, `detail_metrics.csv`, `significance.csv` và `paper_statistics.md`.

Chạy lại dynamic grid cho COAST/VECTRA, MARDAM và các ablation local rồi tự sinh report:

```bash
DATASETS_ROOT=data/datasets/dvrptw_dynamic_grid \
OUTPUT_ROOT=output/dynamic_benchmark_raw \
bash script/run_dynamic_experiment_matrix.sh
```

Sinh và đánh giá các tập OOD:

```bash
PYTHONPATH=. python script/generate_ood_sets.py \
  --output-dir data/test_sets \
  --batch-size 500

PYTHONPATH=. python script/run_ood_experiments.py \
  --datasets-dir data/test_sets \
  --output-dir output/ood_eval \
  --models vectra,mardam,b0,b1,b3,b5,edgeoff
```

Phân tích diagnostics theo bước sau khi inference với `--save-step-diagnostics`:

```bash
PYTHONPATH=. python script/analyze_behavior_diagnostics.py \
  --input-root output/diagnostics \
  --output-dir output/behavior_analysis
```

Tổng hợp bảng giả thiết H1-H4:

```bash
PYTHONPATH=. python script/build_hypothesis_tables.py \
  --master-summary output/dynamic_benchmark_verified/paper_ready/master_summary.csv \
  --ood-summary output/ood_eval/ood_summary.csv \
  --behavior-summary output/behavior_analysis/hypothesis_behavior_summary.csv \
  --output-dir output/hypothesis_tables
```

Hoặc chạy toàn bộ pipeline H1-H4 bằng một script:

```bash
DATASETS_ROOT=/path/to/dvrptw_dynamic_grid \
bash script/run_hypothesis_experiments.sh
```

Runbook đầy đủ cho phần thực nghiệm nằm ở `paper/EXPERIMENT_RUNBOOK.md`.

## Cài Đặt

Core dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` hiện gồm:

```text
torch
scipy
matplotlib
tqdm
```

Một số script phụ có thể cần package bổ sung, ví dụ `openpyxl` cho export `.xlsx` trong batch inference.

## Ghi Chú Về Tính Chính Xác Của Tài Liệu

- Tên nghiên cứu: **COAST**.
- Tên code/model class: **VECTRA**.
- Full model trong CLI: `--ablation-profile vectra`.
- `LookaheadHead` hiện học implicit qua policy gradient, không có loss MSE/TD riêng.
- `OwnershipHead` hiện là latent bias, không có ground-truth assignment supervision.
- `FleetEncoder` trong forward hiện encode acting vehicle với customer context; multi-vehicle coordination nằm ở memory/ownership.
