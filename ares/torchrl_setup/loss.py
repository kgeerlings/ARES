import torch
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from ares.torchrl_setup.hyperparameters_and_setup import (
    clip_epsilon,
    entropy_coef,
    critic_coef,
    loss_critic_type,
    normalize_advantage,
    use_entropy_loss,
    gamma,
    lmbda,
    max_grad_norm,
    learning_rate,
)
from ares.torchrl_setup.policy import policy
from ares.torchrl_setup.critic import critic


loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_coef,
    critic_coef=critic_coef,
    loss_critic_type=loss_critic_type,
    normalize_advantage=normalize_advantage,
    entropy_bonus=use_entropy_loss,
)
loss_module.set_keys(  # We have to tell the loss where to find the keys
    reward="reward",
    action="action",
    sample_log_prob="sample_log_prob",
    value="state_value",
    value_target="value_target",
    done="done",
    terminated="terminated",
)
print("loss_module: ", loss_module)
print("loss_module keys: ", loss_module.in_keys)


loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)

max_grad_norm = max_grad_norm

optimizer = torch.optim.Adam(loss_module.parameters(), learning_rate)
