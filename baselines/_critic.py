from baselines._base import Baseline

import torch
import torch.nn as nn


class CriticBaseline(Baseline):
    """
    Actor-Critic baseline.

    The value network takes as input the current *encoded state*:
        [veh_repr  ||  mean(cust_repr)]   shape: (N, 1, 2 * model_size)

    This avoids the circular dependency of the old design (which fed raw
    policy logits / compat scores into the critic) and gives the critic
    a stable, policy-independent view of the environment state.

    Falls back to the legacy logit-based input when the learner does not
    expose a ``model_size`` attribute (e.g. the original AttentionLearner).
    """

    def __init__(self, learner, cust_count, use_qval = True, use_cumul_reward = False, hidden_size = None):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval   # kept for API compatibility; ignored in state-critic mode

        if hidden_size is None:
            hidden_size = 256

        model_size = getattr(learner, 'model_size', None)
        if model_size is not None:
            # State-based critic: richer, policy-independent input
            input_size = model_size * 2
            self.use_state_critic = True
        else:
            # Legacy fallback: raw compatibility scores
            input_size = cust_count + 1
            self.use_state_critic = False

        # Deeper network with normalisation for faster, more stable learning
        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_step_state(self, veh_repr):
        """Compute V(s) from encoded vehicle + mean-customer context.

        :param veh_repr:   (N, 1, model_size) – current vehicle encoding
        :return:           (N, 1) value estimate
        """
        # Detach actor tensors: critic value loss must not flow gradients
        # back through the actor's computation graph (they are separate networks).
        cust_context = self.learner.cust_repr.detach().mean(dim = 1, keepdim = True)  # (N, 1, D)
        state = torch.cat([veh_repr.detach(), cust_context], dim = -1)                 # (N, 1, 2D)
        val = self.project(state)                                                       # (N, 1, 1)
        return val.squeeze(-1)                                                          # (N, 1)

    def _eval_step_legacy(self, vrp_dynamics, learner_compat, cust_idx):
        """Original logit-based value estimate (fallback for old learners)."""
        compat = learner_compat.clone()
        compat[vrp_dynamics.cur_veh_mask] = 0
        val = self.project(compat)
        if self.use_qval:
            val = val.gather(2, cust_idx.unsqueeze(1).expand(-1, 1, -1))
        return val.squeeze(1)

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx, veh_repr = None):
        if self.use_state_critic:
            # Reuse veh_repr already computed in __call__ loop — avoids a
            # second FleetEncoder forward pass per step.
            if veh_repr is None:
                veh_repr = self.learner._repr_vehicle(
                    vrp_dynamics.vehicles,
                    vrp_dynamics.cur_veh_idx,
                    vrp_dynamics.mask,
                )
            return self._eval_step_state(veh_repr)
        return self._eval_step_legacy(vrp_dynamics, learner_compat, cust_idx)

    # ------------------------------------------------------------------

    def __call__(self, vrp_dynamics):
        vrp_dynamics.reset()
        if hasattr(vrp_dynamics, "veh_speed"):
            self.learner.veh_speed = vrp_dynamics.veh_speed
        if hasattr(self.learner, "_reset_memory"):
            self.learner._reset_memory(vrp_dynamics)
        actions, logps, rewards, bl_vals = [], [], [], []
        while not vrp_dynamics.done:
            if vrp_dynamics.new_customers:
                self.learner._encode_customers(vrp_dynamics.nodes, getattr(vrp_dynamics, 'cust_mask', None))
            cust_idx, logp, veh_repr = self.learner.step(vrp_dynamics)

            # Critic value estimate
            if not(self.use_cumul and bl_vals):
                if self.use_state_critic:
                    bl_vals.append( self._eval_step_state(veh_repr) )
                else:
                    bl_vals.append( self._eval_step_legacy(vrp_dynamics, None, cust_idx) )

            actions.append( (vrp_dynamics.cur_veh_idx, cust_idx) )
            logps.append( logp )
            rewards.append( vrp_dynamics.step(cust_idx) )
        if self.use_cumul:
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = bl_vals[0]
        return actions, logps, rewards, bl_vals

    def parameters(self):
        return self.project.parameters()

    def state_dict(self):
        return self.project.state_dict()

    def load_state_dict(self, state_dict):
        return self.project.load_state_dict(state_dict)

    def to(self, device):
        self.project.to(device = device)
