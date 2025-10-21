import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal
from ares.torchrl_setup.environment import env
from ares.torchrl_setup.hyperparameters_and_setup import num_cells, device

# Device setup

policy_net = nn.Sequential(
    nn.Linear(env.observation_spec["observation"].shape[-1], num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, env.full_action_spec["action"].shape[-1] * 2, device=device),
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=["observation"],
    out_keys=["params"],
)


class SplitParams(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, params):
        loc = params[..., : self.action_dim]
        scale = torch.nn.functional.softplus(params[..., self.action_dim :]) + 1e-4
        return loc, scale


split_module = TensorDictModule(
    SplitParams(env.full_action_spec["action"].shape[-1]),
    in_keys=["params"],
    out_keys=["loc", "scale"],
)
from tensordict.nn import TensorDictSequential

# Combiner les modules
combined_module = TensorDictSequential(policy_module, split_module)


# Build probabilistic actor
policy = ProbabilisticActor(
    module=combined_module,
    in_keys=["loc", "scale"],
    out_keys=["action", "sample_log_prob"],
    distribution_class=TanhNormal,
    return_log_prob=True,
    spec=env.action_spec,
)
