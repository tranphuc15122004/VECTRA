# SBG-MARL: Scale-Balanced Graph Multi-Agent Reinforcement Learning

> Tài liệu kỹ thuật chi tiết về cơ chế **SBG (Scale-Balanced Gating)** và framework **MARL (Multi-Agent Reinforcement Learning)** được triển khai trong VECTRA.  
> Tham khảo kiến trúc tổng thể tại [VECTRA.md](VECTRA.md).

---

## Mục lục

1. [Tổng quan SBG-MARL](#1-tổng-quan-sbg-marl)
2. [Cấu trúc MARL: sMAMDP](#2-cấu-trúc-marl-smamadp)
   - 2.1 [Định nghĩa bài toán](#21-định-nghĩa-bài-toán)
   - 2.2 [Chiến lược chia sẻ policy](#22-chiến-lược-chia-sẻ-policy)
   - 2.3 [Cơ chế phối hợp ngầm](#23-cơ-chế-phối-hợp-ngầm)
3. [SBG: Scale-Balanced Gating](#3-sbg-scale-balanced-gating)
   - 3.1 [Động lực và vấn đề bậc hai](#31-động-lực-và-vấn-đề-bậc-hai)
   - 3.2 [Cheap Score Function](#32-cheap-score-function)
   - 3.3 [Adaptive-K Selection](#33-adaptive-k-selection)
   - 3.4 [Phân tích recall và chất lượng](#34-phân-tích-recall-và-chất-lượng)
4. [MoE: Mixture of Experts](#4-moe-mixture-of-experts)
   - 4.1 [Hai chuyên gia và regime kích hoạt](#41-hai-chuyên-gia-và-regime-kích-hoạt)
   - 4.2 [Uncertainty-Gated Blending](#42-uncertainty-gated-blending)
   - 4.3 [Phân tích toán học](#43-phân-tích-toán-học)
5. [Score Fusion Pipeline](#5-score-fusion-pipeline)
   - 5.1 [Ba nguồn điểm số](#51-ba-nguồn-điểm-số)
   - 5.2 [Z-normalisation](#52-z-normalisation)
   - 5.3 [Fusion MLP](#53-fusion-mlp)
6. [Coordination Memory](#6-coordination-memory)
   - 6.1 [GRU-like Update](#61-gru-like-update)
   - 6.2 [Ownership Head](#62-ownership-head)
   - 6.3 [Vai trò trong MARL](#63-vai-trò-trong-marl)
7. [Training: REINFORCE vs PPO](#7-training-reinforce-vs-ppo)
   - 7.1 [REINFORCE với các baseline](#71-reinforce-với-các-baseline)
   - 7.2 [PPO với GAE](#72-ppo-với-gae)
   - 7.3 [So sánh hai phương pháp](#73-so-sánh-hai-phương-pháp)
8. [Phân tích độ phức tạp](#8-phân-tích-độ-phức-tạp)
9. [Flowchart tổng thể](#9-flowchart-tổng-thể)
10. [Cấu hình và điều chỉnh](#10-cấu-hình-và-điều-chỉnh)

---

## 1. Tổng quan SBG-MARL

**SBG-MARL** là sự kết hợp của hai đổi mới kỹ thuật chính:

| Thành phần | Mục tiêu | Cơ chế |
|---|---|---|
| **SBG** (Scale-Balanced Gating) | Giảm chi phí tính toán từ $O(L_c^2)$ xuống $O(L_c)$ | Lọc ứng viên bằng cheap score → deep scoring chỉ trên $K \ll L_c$ node |
| **MARL** (Multi-Agent RL) | Phối hợp giữa các xe mà không cần communication | CoordinationMemory + OwnershipHead → implicit coordination |
| **MoE** (Mixture of Experts) | Thích nghi với độ khó instance động | Uncertainty-gated expert blending theo entropy/margin |

**Nguyên tắc cốt lõi — "Compute Where It Matters":**

```
Toàn bộ Lc khách ──► [SBG Filter O(Lc)] ──► K ứng viên tốt nhất
                                                      │
                                          [Deep Scoring O(Kd)]
                                                      │
                                          [MoE: tinh chỉnh khi cần]
                                                      │
                                              Quyết định cuối
```

---

## 2. Cấu trúc MARL: sMAMDP

### 2.1 Định nghĩa bài toán

DVRPTW được mô hình hóa như **Sequential Multi-agent Markov Decision Process (sMAMDP)** — một lớp đặc biệt của MAMDP nơi các agent hành động tuần tự thay vì đồng thời.

**Không gian trạng thái** $s_t$:

$$s_t = \underbrace{(\mathbf x_{v_1},\dots,\mathbf x_{v_M})}_{\text{vị trí xe}} \cup \underbrace{(q_{v_1},\dots,q_{v_M})}_{\text{sức chứa}} \cup \underbrace{(t_{v_1},\dots,t_{v_M})}_{\text{thời gian hiện tại}} \cup \underbrace{\mathcal C_t}_{\text{khách hàng đã biết}}$$

**Không gian hành động** $\mathcal A_t$: với xe đang hành động $v^*$:

$$a_t \in \mathcal C_t^{\text{feasible}}(v^*) \cup \{\text{depot}\}$$

trong đó "feasible" nghĩa là phục vụ $c$ không vi phạm capacity và time window của $v^*$.

**Hàm reward:**

$$r_t = -d(x_{v^*}, x_{a_t}) - \lambda_{\text{late}} \cdot \max(0,\; t_{\text{arrival}} - l_{a_t})$$

**Transition:** Sau khi $v^*$ chọn $a_t$, lịch sử của $v^*$ được cập nhật; xe kế tiếp với thời gian nhỏ nhất trở thành acting vehicle.

### 2.2 Chiến lược chia sẻ policy

Thay vì học $M$ policy riêng biệt (tốn bộ nhớ và khó tổng quát hóa), VECTRA dùng **một policy chia sẻ** $\pi_\theta$ cho tất cả xe:

$$\pi_\theta(a \mid s_t, v^*) = \text{softmax}\!\left(\text{compat}(v^*, \mathcal C_t^{\text{feasible}})\right)$$

**Lợi ích:**
- Số tham số không phụ thuộc vào $M$
- Xe mới thêm vào không cần huấn luyện lại
- Dữ liệu học hiệu quả hơn — mỗi bước cập nhật đóng góp kinh nghiệm cho tất cả xe

**Điều kiện đủ để policy chia sẻ hoạt động:** Biểu diễn trạng thái $s_t$ phải phân biệt được $v^*$ với các xe khác — được đảm bảo bởi `veh_idx` và CoordinationMemory riêng cho từng xe.

### 2.3 Cơ chế phối hợp ngầm

Trong sMAMDP, xe $v^*$ không thấy kế hoạch của $v_{-v^*}$. Thay vào đó, phối hợp được thực hiện ngầm qua:

```
Xe v1 chọn c3 ──► CoordinationMemory[v1] cập nhật
                         │
                         ▼
                  OwnershipHead đọc toàn bộ memory
                         │
                         ▼
              O[v2, c3] giảm ──► xe v2 ít likely chọn c3
```

Đây là dạng **implicit communication qua shared state** — không cần message passing giữa agent.

---

## 3. SBG: Scale-Balanced Gating

### 3.1 Động lực và vấn đề bậc hai

Trong kiến trúc attention thuần túy, scoring $L_c$ khách hàng tốn:

$$T_{\text{dense-score}} = O(L_c \cdot d) \quad \text{cho mỗi bước}$$

Nhưng $L_c$ có thể lên tới 100–200, và mỗi episode có $O(M L_c)$ bước, nên tổng chi phí $O(M L_c^2 d)$ trở nên nặng. Hơn nữa:

- Phần lớn $L_c$ khách **không khả thi** (vi phạm TW hoặc capacity)
- Phần lớn khách khả thi **rõ ràng không tốt** (quá xa, quá trễ)

**Insight:** Chỉ cần tìm đúng top-$K$ ứng viên — không cần score chính xác cho tất cả.

### 3.2 Cheap Score Function

SBG tính một **cheap scalar score** cho mỗi khách chỉ từ precomputed scalars (không qua neural network):

$$\boxed{s_c^{\text{cheap}} = -d_{v^*c} \;-\; \lambda_{\text{late}} \cdot \ell_{v^*c} \;+\; \mu_{\text{slack}} \cdot \sigma_{v^*c} \;+\; \omega_{\text{own}} \cdot O_{v^*,c}}$$

| Số hạng | Ý nghĩa vật lý | Giá trị mặc định |
|---|---|---|
| $-d_{v^*c}$ | Ưu tiên khách gần (giảm quãng đường) | weight = 1 |
| $-\lambda_{\text{late}} \cdot \ell_{v^*c}$ | Phạt nếu arrive trễ so với deadline | $\lambda = 2.0$ |
| $+\mu_{\text{slack}} \cdot \sigma_{v^*c}$ | Bonus nếu time window còn rộng | $\mu = 0.5$ |
| $+\omega_{\text{own}} \cdot O_{v^*,c}$ | Bonus nếu xe này đang "sở hữu" khách | $\omega = 0.5$ |

Các thành phần:

$$d_{v^*c} = \|x_{v^*} - x_c\|_2$$

$$\ell_{v^*c} = \max(0,\; t_{v^*} + \tau_{v^*c} - l_c), \quad \tau_{v^*c} = d_{v^*c} / \text{speed}$$

$$\sigma_{v^*c} = \max(0,\; l_c - (t_{v^*} + \tau_{v^*c}))$$

$$O_{v^*,c} = \text{OwnershipHead}(\mathbf m_{v^*}, \mathbf h_c) \in [0,1]$$

**Tại sao dùng $O_{v^*,c}$ trong cheap score?**  
Ownership prior được tính **trước SBG** (chỉ cần dot-product giữa memory vector và customer embedding), nên chi phí $O(L_c d)$ là chấp nhận được. Đưa nó vào cheap score giúp filter giữ lại ứng viên mà $v^*$ đang "theo đuổi", tăng recall.

**Xử lý all-masked (edge case):** Nếu tất cả khách đều infeasible, SBG unlock khách đầu tiên ($c_0$) để tránh `nan` trong softmax — xe sẽ quay về depot.

### 3.3 Adaptive-K Selection

Thay vì dùng $K$ cố định, SBG điều chỉnh thích nghi dựa trên **mật độ ứng viên khả thi**:

$$r_f = \frac{|\{c : \text{feasible}(v^*, c)\}|}{L_c}$$

| $r_f$ | Điều chỉnh $K$ | Lý do |
|---|---|---|
| $r_f > 0.6$ | $K \leftarrow \lfloor 1.5K \rfloor$ | Nhiều lựa chọn tốt → cần shortlist rộng hơn |
| $r_f < 0.3$ | $K \leftarrow \lfloor 0.75K \rfloor$ | Khan hiếm → cheap score đã đủ phân biệt |
| $0.3 \le r_f \le 0.6$ | $K$ giữ nguyên | Tình huống bình thường |

Sau điều chỉnh, $K$ được clamp:

$$K \leftarrow \text{clamp}\!\left(K,\; K_{\min},\; \min(K_{\max}, L_c)\right)$$

**Sơ đồ adaptive-K:**

```
INPUT: feasible_count, Lc, K_base
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│        Tính r_f = feasible_count / Lc                   │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │       r_f > 0.6 ?        │
              └────┬──────────────┬──────┘
                Yes│              │No
                   ▼              ▼
           K ← ⌊1.5K⌋     r_f < 0.3 ?
                          ┌──────┴──────┐
                        Yes│             │No
                           ▼             ▼
                    K ← ⌊0.75K⌋    K giữ nguyên
                           │             │
                           └──────┬──────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────┐
│    clamp K ∈ [K_min, min(K_max, Lc)]                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│    topK(cheap_score) → cand_idx  (N × K)                │
└─────────────────────────────────────────────────────────┘
```

### 3.4 Phân tích recall và chất lượng

**Bổ đề (SBG recall).** Gọi $a^*$ là hành động tối ưu theo deep scorer. Cheap score tương quan với deep score qua:

$$s_c^{\text{cheap}} \approx -d_{vc} + f(TW_c, O_{v^*,c}) \approx \alpha \cdot q_c + \beta + \epsilon_c$$

với $\epsilon_c$ là noise zero-mean. Khi $K \ge K_{\min}$ và $q_{a^*} - q_{(K+1)} \ge \gamma_{\text{margin}}$, xác suất recall:

$$P(a^* \in \mathcal S_K) \ge 1 - \frac{\text{Var}(\epsilon)}{\gamma_{\text{margin}}^2} \cdot \frac{1}{K}$$

Điều này giảm khi $K$ tăng và khi $\gamma_{\text{margin}}$ tăng (instance dễ phân biệt hơn).

**Ràng buộc chất lượng rollout.** Với miss rate $\delta = P(a^* \notin \mathcal S_K)$, từ Proposition §3.2 trong VECTRA.md:

$$\mathbb{E}[\Delta J] \le T \cdot \delta \cdot \Delta_{\max}$$

Tăng $K$ giảm $\delta$, nhưng tăng chi phí $O(Kd)$ — đây là **trade-off chính** của SBG.

---

## 4. MoE: Mixture of Experts

### 4.1 Hai chuyên gia và regime kích hoạt

SBG-MARL dùng **hai chuyên gia phần mềm** (soft experts) thay vì routing cứng:

**Expert Local (myopic)** — ưu tiên quyết định ngắn hạn:
$$e_{\text{local},c} = 0.7\tilde s_{\text{att}} + 0.9\tilde s_{\text{own}} - 0.2\tilde s_{\text{look}}$$

*Nhấn mạnh att-score và ownership, hạ thấp lookahead — phù hợp khi supply khan hiếm và cần serve ngay.*

**Expert Future (anticipatory)** — ưu tiên tương lai:
$$e_{\text{future},c} = 0.4\tilde s_{\text{att}} + 0.3\tilde s_{\text{own}} - 0.9\tilde s_{\text{look}}$$

*Nhấn mạnh lookahead âm (penalise bad future) — phù hợp khi có nhiều lựa chọn và cần tránh dead-ends.*

**Regime kích hoạt:**

```python
use_local  = feasible_ratio < 0.35          # khan hiếm ứng viên
use_future = (look_std > 0.9) & (~use_local) # lookahead phân tán cao
```

| Condition | Expert kích hoạt | Lý do |
|---|---|---|
| $r_f < 0.35$ | Local | Phải quyết định nhanh, ít lựa chọn |
| $\sigma_{\text{look}} > 0.9$ | Future | Tương lai không chắc, penu lookahead |
| Cả hai không | Chỉ base | Tình huống rõ ràng, không cần expert |

### 4.2 Uncertainty-Gated Blending

MoE **không luôn luôn** kích hoạt — nó chỉ can thiệp khi model đang uncertain:

**Bước 1: Đo lường uncertainty**

$$H_t = -\sum_{c \in \mathcal S_K} p_c \log p_c, \quad p_c = \text{softmax}(\text{compat\_base})_c$$

$$\Delta_t = \text{compat\_base}_{(1)} - \text{compat\_base}_{(2)} \quad \text{(margin top-2)}$$

$$\bar H_t = \frac{H_t}{\log |\mathcal S_K|} \in [0, 1] \quad \text{(entropy chuẩn hóa)}$$

**Bước 2: Xác định MoE có active không**

$$\text{MoE\_active} = \mathbf 1\!\left[\bar H_t > H_{\text{floor}} \;\wedge\; \Delta_t < \Delta_{\text{ceil}}\right]$$

Nếu model đã confident ($\bar H_t$ thấp hoặc $\Delta_t$ cao) → MoE tắt, giữ nguyên base score.

**Bước 3: Dynamic strength**

$$\alpha_t = \left(\alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{\bar H_t}{1 + \Delta_t}\right) \cdot \text{MoE\_active}$$

$$\alpha_{\text{local},t} = \alpha_t \cdot \mathbf 1[\text{use\_local}], \quad \alpha_{\text{fut},t} = \alpha_t \cdot \mathbf 1[\text{use\_future}]$$

**Bước 4: Blending**

$$\delta_* = \text{clamp}(e_* - \text{compat\_base},\; -2,\; 2)$$

$$\text{compat}_c = \text{compat\_base}_c + \alpha_{\text{local}} \cdot \delta_{\text{local},c} + \alpha_{\text{fut}} \cdot \delta_{\text{fut},c}$$

### 4.3 Phân tích toán học

**Tại sao clamp $\delta$ vào $[-2, 2]$?**  
Tránh expert overwrite hoàn toàn base score. MoE chỉ **điều chỉnh** (nudge) phân phối, không thay thế.

**Tại sao $\alpha \le 0.15$?**  
Với $\alpha$ nhỏ, perturbation cực đại lên logit là $0.15 \times 2 = 0.3$ đơn vị. Sau tanh cap ($C=10$), đây là thay đổi xác suất $\approx 3\%$ — đủ để tác động nhưng không destabilize policy.

**Bổ đề (MoE không làm tổi nghiệm rõ ràng).** Gọi $a^*$ là argmax của `compat_base`. Nếu $\gamma = \text{compat\_base}_{a^*} - \max_{c \ne a^*} \text{compat\_base}_c > 4\alpha_{\max}$, thì $a^*$ vẫn là argmax sau MoE blending.

*Chứng minh:* $\text{compat}_{a^*} \ge \text{compat\_base}_{a^*} - \alpha_{\max} \cdot 2$. Với $c \ne a^*$: $\text{compat}_c \le \text{compat\_base}_c + \alpha_{\max} \cdot 2$. Vậy $\text{compat}_{a^*} - \text{compat}_c \ge \gamma - 4\alpha_{\max} > 0$. $\square$

---

## 5. Score Fusion Pipeline

### 5.1 Ba nguồn điểm số

Sau khi có shortlist $\mathcal S_K$, deep scoring được thực hiện qua ba module song song:

```
                     ┌──────────────────┐
   veh_repr (N×1×D)  │  CrossEdgeFusion │──► att_score   (N×1×K)
   cust_repr (N×K×D) │  Multi-head att  │
   edge_emb  (N×1×K×D)└──────────────────┘

                     ┌──────────────────┐
   veh_memory(N×Lv×H)│  OwnershipHead   │──► owner_bias  (N×1×K)
   cust_repr (N×K×D) │  Bilinear score  │
                     └──────────────────┘

                     ┌──────────────────┐
   veh_repr  (N×1×D) │  LookaheadHead   │──► look_score  (N×1×K)
   cust_repr (N×K×D) │  MLP concat feat │
   edge_emb  (N×1×K×D)└──────────────────┘
```

**CrossEdgeFusion** — semantic + geometric score:
$$s_{\text{att},c} = \frac{1}{H}\sum_h \left(\frac{\mathbf q^{(h)} \cdot \mathbf k_c^{(h)}}{\sqrt{d_h}} + (W_e \mathbf e_{v^*c})^{(h)}\right)$$

**OwnershipHead** — coordination prior:
$$s_{\text{own},c} = \log \text{softmax}\!\left(\frac{W_v \mathbf m_{v^*} \cdot (W_c \mathbf H^{\text{cust}}_c)^\top}{\sqrt{D}}\right)$$

**LookaheadHead** — future value estimate:
$$s_{\text{look},c} = \text{MLP}([\mathbf h_{v^*}^{\text{veh}};\, \mathbf h_c^{\text{cust}};\, \mathbf e_{v^*c}])$$

### 5.2 Z-normalisation

Ba scores có **phân phối và scale khác nhau** (att-score ~ logit không bị chặn; owner_bias ~log-prob âm; look ~giá trị tùy ý). Chuẩn hóa Z đảm bảo đóng góp công bằng:

$$\tilde s_{\bullet,c} = \frac{s_{\bullet,c} - \frac{1}{K}\sum_{c'} s_{\bullet,c'}}{\sqrt{\frac{1}{K}\sum_{c'}(s_{\bullet,c'} - \bar s_\bullet)^2} + 10^{-8}}$$

**Lập luận:** Nếu một nguồn có phương sai cực cao (ví dụ att-score spread $\pm 5$) nhưng nguồn khác chỉ $\pm 0.1$, Fusion MLP sẽ bị dominated bởi nguồn đầu mà không học được từ nguồn sau. Z-norm giải quyết điều này.

### 5.3 Fusion MLP

$$[\tilde s_{\text{att},c};\, \tilde s_{\text{own},c};\, \tilde s_{\text{look},c}] \xrightarrow{W_1 \in \mathbb R^{64\times 3}} \text{ReLU} \xrightarrow{W_2 \in \mathbb R^{1\times 64}} \text{compat\_base}_c$$

MLP học **tương tác phi tuyến** giữa ba nguồn — ví dụ: "att cao + look thấp" có thể tốt hơn "att trung bình + look trung bình" trong regime khan hiếm.

**Tanh exploration cap:**

$$\tilde{\text{compat}}_c = C \cdot \tanh\!\left(\frac{\text{compat}_c}{C}\right), \quad C = 10$$

Tác dụng: Logit bị giới hạn trong $(-C, C)$, tránh policy collapse sang deterministic quá sớm. Gradient luôn non-zero tại mọi giá trị.

---

## 6. Coordination Memory

### 6.1 GRU-like Update

`CoordinationMemory` duy trì một hidden state $\mathbf m_v \in \mathbb R^H$ cho mỗi xe, hoạt động như GRU đơn giản (không reset gate):

**Sau khi xe $v^*$ được gán khách $c^*$:**

$$\mathbf x_t = [\mathbf h_{v^*}^{\text{veh}};\; \mathbf h_{c^*}^{\text{cust}};\; \mathbf e_{v^*,c^*}] \in \mathbb R^{3D}$$

$$\mathbf m_{v^*}^{t+1} = \tanh\!\left(W_x \mathbf x_t + W_h \mathbf m_{v^*}^t\right)$$

Các xe khác $v \ne v^*$ **không thay đổi memory** tại bước này.

**Triển khai hiệu quả** — dùng `scatter` thay vì for-loop:

```python
# Chỉ cập nhật slot của v* trong memory tensor N×Lv×H
updated = memory.scatter(
    dim=1,
    index=veh_idx[:, :, None].expand(-1, -1, memory.size(-1)),
    src=next_h.unsqueeze(1)
)
```

Phức tạp: $O(1)$ per step (không phụ thuộc $M$).

### 6.2 Ownership Head

Ownership Head đọc toàn bộ memory $\mathbf M \in \mathbb R^{N \times L_v \times H}$ để tính ma trận soft-assignment giữa mọi xe và mọi khách:

$$O_{v,c} = \text{softmax}_v\!\left(\frac{(W_v \mathbf m_v) \cdot (W_c \mathbf h_c)^\top}{\sqrt{D}}\right) \in [0,1]^{L_v \times L_c}$$

Chú ý `softmax` trên chiều $v$ (vehicle dimension): $\sum_v O_{v,c} = 1$ với mọi $c$.

Ý nghĩa: $O_{v,c}$ là xác suất "xe $v$ nên serve $c$" dựa trên lịch sử quyết định.

**Log-prior cho xe đang hành động:**

$$\text{owner\_bias}_{v^*,c} = \log O_{v^*,c} \le 0$$

Đây là **prior âm** — khách mà xe $v^*$ chưa "claim" sẽ có prior thấp hơn.

### 6.3 Vai trò trong MARL

**Tại sao memory giúp coordination mà không cần communication?**

Xem xét kịch bản: Xe $v_1$ vừa chọn cụm khách ở góc đông-bắc. Memory $\mathbf m_{v_1}$ được cập nhật với embedding của các khách đó.

- $O_{v_1, c_{\text{NE}}}$ tăng (v1 có xu hướng serve NE cluster)
- $O_{v_2, c_{\text{NE}}}$ vẫn thấp

Khi $v_2$ đến lượt, `owner_bias` cộng thêm $\log O_{v_2, c_{\text{NE}}} < 0$ → xe $v_2$ bị discourage chọn cụm NE, tự nhiên hướng sang vùng khác.

**Đây là emergent coordination** — không encode rule cứng, learn từ reward.

---

## 7. Training: REINFORCE vs PPO

VECTRA hỗ trợ hai paradigm training riêng biệt.

### 7.1 REINFORCE với các baseline

Được triển khai trong [MODEL/train.py](MODEL/train.py).

**Objective:**

$$\mathcal L_{\text{REINFORCE}} = -\frac{1}{N}\sum_{i=1}^N \sum_{t=1}^{T_i} \left(R_t^{(i)} - b(s_t^{(i)})\right) \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})$$

**Các baseline được hỗ trợ:**

| Baseline | $b(s_t)$ | Ưu điểm | Nhược điểm |
|---|---|---|---|
| `none` | $0$ | Đơn giản | Variance cao |
| `nearnb` | Cost nearest-neighbour | Không trainable | Yếu với dynamic |
| `rollout` | Cost của greedy rollout | Adapted theo training | Tốn compute |
| `critic` | $V_\phi(s_t)$ (learned) | Variance thấp | Phải train thêm critic |

**Critic baseline** dùng architecture riêng đọc `cust_repr` và tính giá trị trạng thái.

**Cumulative reward** (loss_use_cumul):

$$R_t = \sum_{t'=t}^{T} r_{t'} \quad \text{(return từ bước } t \text{)}$$

vs sum-of-rewards:

$$R_t = \sum_{t'=1}^{T} r_{t'} \quad \text{(tổng toàn episode)}$$

### 7.2 PPO với GAE

Được triển khai trong [MODEL/train_PPO.py](MODEL/train_PPO.py).

**Collect rollout** — chạy episode một lần, lưu lại:
- `old_logps`: log probability của actions thực sự chọn
- `rewards_t`: reward tại mỗi bước
- `values_t`: $V_\phi(s_t)$ từ critic

**GAE (Generalized Advantage Estimation):** Thay vì dùng $R_t - V(s_t)$ thuần tuý, GAE kết hợp TD errors theo exponential weighting:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Trong code, tính ngược từ cuối episode:

```python
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * next_value - values[t]
    gae   = delta + gamma * lam * gae
    advantages[t] = gae
    returns[t]    = gae + values[t]
    next_value    = values[t]
```

Với $\lambda \in [0,1]$: $\lambda=0$ → TD(0) (low variance, high bias); $\lambda=1$ → Monte Carlo (high variance, low bias).

**PPO clip objective:**

$$\mathcal L_{\text{clip}} = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\right)\right]$$

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$$

**Value loss** (smooth L1):

$$\mathcal L_{\text{value}} = \text{SmoothL1}(V_\phi(s_t),\; R_t)$$

**Entropy bonus** (khuyến khích exploration):

$$\mathcal L_{\text{entropy}} = -\mathbb{E}_t\!\left[\sum_c \pi_\theta(c \mid s_t) \log \pi_\theta(c \mid s_t)\right]$$

**Tổng loss PPO:**

$$\mathcal L_{\text{PPO}} = \mathcal L_{\text{clip}} + c_v \mathcal L_{\text{value}} - c_e \mathcal L_{\text{entropy}}$$

**Early stopping bằng KL divergence:** Nếu approximate KL vượt ngưỡng `ppo_target_kl`, dừng update epoch ngay:

```python
approx_kl = (old_logps - new_logps).mean()
if approx_kl > ppo_target_kl:
    break
```

**Advantage normalisation** (tuỳ chọn, `ppo_adv_norm`):

$$\hat A_t \leftarrow \frac{\hat A_t - \mu_A}{\sigma_A + 10^{-8}}$$

### 7.3 So sánh hai phương pháp

| Tiêu chí | REINFORCE | PPO |
|---|---|---|
| **Sample efficiency** | Thấp (mỗi sample dùng 1 lần) | Cao (dùng lại `ppo_epochs` lần) |
| **Stability** | Dễ high variance | Ổn định hơn nhờ clip ratio |
| **Implementation** | Đơn giản | Phức tạp hơn (collect → evaluate loop) |
| **Baseline/Critic** | Cần train song song | Critic integral (CriticBaseline) |
| **Phù hợp với** | Prototype nhanh | Training dài, large-scale |
| **SBG-MARL preset** | `--sbg_train_ready` | `--sbg_train_ready` |

**Khuyến nghị:**
- Thử nghiệm nhanh → REINFORCE + rollout baseline
- Production training → PPO với `--ppo_epochs 3 --ppo_gae_lambda 0.95`

---

## 8. Phân tích độ phức tạp

### Per-step complexity

| Module | Không có SBG | Có SBG ($K \ll L_c$) |
|---|---|---|
| Customer encoding (cached) | $O(\bar D L_c k d)$ | $O(\bar D L_c k d)$ (giống) |
| Edge feature build | $O(L_c)$ | $O(L_c)$ |
| Cheap score filter | — | $O(L_c)$ |
| CrossEdgeFusion | $O(L_c d)$ | $O(K d)$ |
| OwnershipHead | $O(L_v L_c)$ | $O(L_v K)$ |
| LookaheadHead | $O(L_c d)$ | $O(K d)$ |
| Score Fusion MLP | $O(L_c)$ | $O(K)$ |
| MoE blending | $O(L_c)$ | $O(K)$ |
| Memory update | $O(d)$ | $O(d)$ |
| **Tổng scoring** | **$O(L_c d)$** | **$O(L_c + Kd)$** |

Với $K = 16$, $d = 128$, $L_c = 100$: tiết kiệm ~6× trong scoring phase.
Với $L_c = 200$: tiết kiệm ~12×.

### Complexity theo episode

Một episode có $N_{\text{steps}} = O(M L_c)$ bước (mỗi xe phục vụ $O(L_c/M)$ khách):

| | Không SBG | Có SBG |
|---|---|---|
| Customer encoding | $O(\bar D L_c k d)$ **một lần** (event-driven cache) | Giống |
| Scoring per step | $O(L_c d)$ | $O(L_c + Kd)$ |
| **Tổng episode** | $O(\bar D L_c k d + M L_c \cdot L_c d)$ | $O(\bar D L_c k d + M L_c(L_c + Kd))$ |

Cho $M=5$, $L_c=100$, $d=128$, $K=16$: SBG tiết kiệm khoảng **80% trong scoring**.

---

## 9. Flowchart tổng thể

### 9.1 SBG Pipeline chi tiết

```
INPUT: vehicles, nodes (N×Lc×F), edge_emb, owner_bias, veh_mask
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│          SBG FILTER — O(Lc)                             │
│   Tính d_vc, τ_vc, arrival, ℓ_vc, σ_vc cho mọi c      │
│   s_cheap = -d - λ·ℓ + μ·σ + ω·O                      │
│   masked_fill(infeasible) → -∞                          │
│   r_f = feasible_count / Lc                             │
│                                                          │
│   r_f > 0.6 ──Yes──► K ← ⌊1.5K⌋ ──┐                  │
│      │                               │                   │
│      No                              ▼                   │
│      ▼                    clamp K ∈ [Kmin, Kmax]        │
│   r_f < 0.3 ──Yes──► K ← ⌊0.75K⌋──┘                  │
│      │                                                   │
│      No──────────────► K giữ nguyên ──────────────────► │
│                                                          │
│   topK(cheap_score) → cand_idx  (N × K)                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│          DEEP SCORING — O(Kd)                           │
│   Gather cust_repr, edge_emb, owner_bias trên K node    │
│                                                          │
│   ┌──────────────────┐  ┌───────────────────────────┐   │
│   │  CrossEdgeFusion │  │      LookaheadHead        │   │
│   │  multi-head att  │  │  MLP(veh‖cust‖edge) → 1   │   │
│   │  + edge bias     │  └────────────┬──────────────┘   │
│   │  → att_score     │               │  look_score       │
│   └────────┬─────────┘               │  N×1×K           │
│            │  att_score N×1×K        │                   │
│            └──────────────┬──────────┘                   │
│                           │                              │
│              Z-norm att, own, look                       │
│                           │                              │
│              Fusion MLP  (3 → 64 → 1)                   │
│                           │                              │
│                    compat_base  N×1×K                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│          MOE BLENDING — O(K)                            │
│   H = entropy(softmax(compat_base))                     │
│   Δ = top1_score − top2_score                           │
│                                                          │
│   H > H_floor AND Δ < Δ_ceil ?                          │
│      │Yes                     │No                       │
│      ▼                        ▼                         │
│   α_dynamic                α = 0  (MoE off)             │
│   = αmin + range·H/(1+Δ)                                │
│      │                                                   │
│      ├── r_f < 0.35 ──► blend expert_local              │
│      │                   (0.7·att + 0.9·own − 0.2·look) │
│      └── σ_look > 0.9 ─► blend expert_future            │
│                           (0.4·att + 0.3·own − 0.9·look)│
│                                                          │
│   compat = base + α_local·δ_local + α_fut·δ_fut         │
│   δ = clamp(expert − base, −2, 2)                       │
│                                                          │
│   tanh cap:  C · tanh(compat / C),  C = 10              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│   log_softmax(compat, mask infeasible)                  │
│   scatter logp_local → full Lc dim                      │
│   sample / greedy → cust_idx                            │
└─────────────────────────────────────────────────────────┘
```

### 9.2 MARL Coordination Flow

```
══════════════════════════════════════════════════════════════
  BƯỚC t: Xe v* hành động
══════════════════════════════════════════════════════════════

   memory M  (N×Lv×H)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  OwnershipHead                                          │
│  O = softmax_v ( Wv·M · (Wc·H_cust)ᵀ / √D )           │
│  → O  (N × Lv × Lc)                                    │
│  Gather acting row: owner_bias = log O[v*]  (N×1×Lc)   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  SBG Filter + Deep Scoring + MoE                        │
│  → logp  (N × Lc)                                      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
                   Sample → cust_idx = c*
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  _update_memory  (chỉ slot v* thay đổi)                 │
│  x  = [ h_veh(v*) ; h_cust(c*) ; e(v*,c*) ]            │
│  m_v*_new = tanh( Wx·x + Wh·m_v*_old )                 │
│  M_new = scatter(M, slot=v*, value=m_v*_new)            │
└──────────────────────────┬──────────────────────────────┘
                           │  M_new (slot v* cập nhật,
                           │         các xe khác nguyên)
══════════════════════════════════════════════════════════════
  BƯỚC t+1: Xe v' hành động  (v' ≠ v*)
══════════════════════════════════════════════════════════════
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  OwnershipHead đọc M_new                                │
│  O'[v', c*] thấp hơn vì m_v* đã encode việc chọn c*    │
│  → owner_bias[v', c*] âm hơn                            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  v' bị discourage chọn c* (emergent coordination)       │
│  v' tự nhiên hướng sang cluster / vùng khác             │
└─────────────────────────────────────────────────────────┘
```

### 9.3 PPO Training Loop

```
([Khởi tạo model + optimizer + scaler])
         │
         ▼
╔═════════════════════════════════════════════════════════╗
║                    EPOCH LOOP                           ║
║                                                         ║
║  DataLoader(train_data, batch_size, shuffle=True)       ║
║         │                                               ║
║         ▼                                               ║
║  ┌──────────────────────────────────────────────────┐   ║
║  │  COLLECT ROLLOUT                                 │   ║
║  │  run baseline.forward(dyna)                      │   ║
║  │  → old_logps, rewards, values, actions (detach)  │   ║
║  │                                                  │   ║
║  │  GAE (ngược từ T về 0):                          │   ║
║  │    δt = r_t + γ·V(s_{t+1}) − V(s_t)            │   ║
║  │    gae = δt + γλ·gae                             │   ║
║  │    advantages[t] = gae                           │   ║
║  │    returns[t]    = gae + V(s_t)                  │   ║
║  │                                                  │   ║
║  │  [tuỳ chọn] advantage normalisation              │   ║
║  │    Â ← (Â − μ_A) / (σ_A + ε)                   │   ║
║  └───────────────────────┬──────────────────────────┘   ║
║                          │                              ║
║         ┌────────────────▼──────────────────────┐       ║
║         │  PPO UPDATE — lặp ppo_epochs lần       │       ║
║         │                                        │       ║
║         │  Re-evaluate rollout với θ_current     │       ║
║         │  ratio = exp(logπ_new − logπ_old)      │       ║
║         │                                        │       ║
║         │  L_clip  = −min(ratio·Â, clip·Â)      │       ║
║         │  L_value = SmoothL1(V_φ(s), R)         │       ║
║         │  L_entropy = −Σ p·log p                │       ║
║         │  L = L_clip + c_v·L_value − c_e·L_ent │       ║
║         │                                        │       ║
║         │  backward() → clip_grad → optim.step() │       ║
║         │                                        │       ║
║         │  approx_KL = mean(logπ_old − logπ_new) │       ║
║         │  KL > target_KL? ──Yes──► break epoch  │       ║
║         │       │No                              │       ║
║         │       └──────── lặp lại ───────────────┘       ║
║         └────────────────────────────────────────┘       ║
║                          │                              ║
║                 next minibatch                          ║
╚══════════════════════════╪═════════════════════════════╝
                           │ (sau mỗi epoch)
                           ▼
┌─────────────────────────────────────────────────────────┐
│  lr_sched.step()                                        │
│  pending_cost  *= pend_cost_growth  (nếu có)            │
│  late_cost     *= late_cost_growth  (nếu có)            │
│  max_grad_norm *= grad_norm_decay   (nếu có)            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  (ep+1) % checkpoint_period == 0                        │
│  → save_checkpoint(ep, learner, optim, baseline, sched) │
└──────────────────────────┬──────────────────────────────┘
                           │
                           └──────► (epoch tiếp theo)
```

---

## 10. Cấu hình và điều chỉnh

### Các preset quan trọng

**SBG-Train-Ready** (`--sbg_train_ready`):

```bash
python script/train_mardam.py \
  --sbg_train_ready \
  --problem_type dvrptw \
  --customers_count 100 \
  --vehicles_count 5 \
  --epoch_count 300
```

Tự động bật: `sbg_enable` ($K=16$, adaptive), `adaptive_depth` ($r_{\text{easy}}=0.7$), `latent_bottleneck` ($M=32$), `sbg_moe_enable` ($\alpha_{\max}=0.03$).

**PPO với SBG:**

```bash
python MODEL/train_PPO.py \
  --problem_type dvrptw \
  --customers_count 50 \
  --vehicles_count 3 \
  --sbg_enable \
  --sbg_cand_k 16 \
  --sbg_adaptive_k \
  --sbg_k_min 8 \
  --sbg_moe_enable \
  --ppo_epochs 3 \
  --ppo_clip_range 0.1 \
  --ppo_gae_lambda 0.95 \
  --ppo_value_coef 0.5 \
  --ppo_entropy_coef 0.01
```

### Hướng dẫn tuning SBG

| Tình huống | Điều chỉnh |
|---|---|
| Recall thấp (missed optimal) | Tăng `sbg_cand_k`, tăng `sbg_k_min` |
| Tốc độ chậm dù đã bật SBG | Giảm `sbg_cand_k`, bật `adaptive_depth` |
| MoE gây destabilize | Giảm `sbg_moe_strength`, tăng `sbg_moe_entropy_floor` |
| MoE ít tác dụng | Giảm `sbg_moe_entropy_floor`, giảm `sbg_moe_margin_ceil` |
| Overfitting quy mô nhỏ | Bật `latent_bottleneck`, tăng `latent_min_nodes` |

### Monitoring SBG hiệu quả

Dùng `ForwardProfiler` để xem phân phối thời gian:

```python
learner.reset_forward_profiling()
_, _, _ = learner(test_env)
summary = learner.get_forward_profiling_summary()
# Trả về: [('encode_customers', t, %), ('score_customers', t, %), ...]
```

Nếu `sbg_select` chiếm > 15% tổng thời gian → `sbg_cand_k` quá lớn hoặc $L_c$ quá nhỏ.

---

## Tóm tắt

SBG-MARL giải quyết ba thách thức cốt lõi của DVRPTW học sâu:

1. **Hiệu quả tính toán**: SBG giảm scoring từ $O(L_c d)$ xuống $O(L_c + Kd)$ — gần tuyến tính theo $L_c$.

2. **Phối hợp đa agent**: CoordinationMemory + OwnershipHead tạo implicit coordination không cần message passing, $O(1)$ per step.

3. **Thích nghi với độ khó**: Uncertainty-gated MoE + Adaptive depth — chỉ tốn compute khi instance thực sự khó.

Ba cơ chế này hoạt động cộng hưởng: SBG tập trung compute vào đúng ứng viên, MoE tinh chỉnh khi không chắc, Memory duy trì coordination qua thời gian.

---

*Xem thêm kiến trúc tổng thể tại [VECTRA.md](VECTRA.md). Xem chi tiết layer implementation tại [layers/Mymodel_layers.py](layers/Mymodel_layers.py).*
