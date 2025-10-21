from torch import nn
from torchrl.modules import ValueOperator
from ares.torchrl_setup.hyperparameters_and_setup import num_cells_critic


critic_net = nn.Sequential(
    nn.LazyLinear(num_cells_critic, bias=True),
    nn.Tanh(),
    nn.LazyLinear(num_cells_critic, bias=True),
    nn.Tanh(),
    nn.LazyLinear(1, bias=True),
)
critic = ValueOperator(
    module=critic_net,
    in_keys=["observation"],
    out_keys=["state_value"],
)
