import torch

class Baseline:
    def __init__(self, learner, use_cumul_reward = False):
        self.learner = learner
        self.use_cumul = use_cumul_reward

    def __call__(self, vrp_dynamics):
        if self.use_cumul:
            actions, logps, rewards = self.learner(vrp_dynamics)
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = self.eval(vrp_dynamics)
        else:
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
                bl_vals.append( self.eval_step(vrp_dynamics, None, cust_idx) )
                actions.append( (vrp_dynamics.cur_veh_idx, cust_idx) )
                logps.append( logp )
                r = vrp_dynamics.step(cust_idx)
                rewards.append(r)
        self.update(rewards, bl_vals)
        return actions, logps, rewards, bl_vals

    def eval(self, vrp_dynamics):
        raise NotImplementedError()

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        raise NotImplementedError()

    def update(self, rewards, bl_vals):
        pass

    def parameters(self):
        return []

    def state_dict(self, destination = None):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def to(self, device):
        pass
