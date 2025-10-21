from torchrl.envs import ParallelEnv
from torchrl.collectors import SyncDataCollector

from ares.torchrl_setup.environment import env, create_env
from ares.torchrl_setup.hyperparameters_and_setup import (
    frames_per_batch,
    total_frames,
    device,
)
from ares.torchrl_setup.policy import policy
from ares.torchrl_setup.critic import critic

# Initialize network weights for agent
td_reset = env.reset()
print("policy", policy)
print("td_reset: ", td_reset)
out_keys = "action"
if policy is not None:
    policy(td_reset)
if critic is not None:
    critic(td_reset)
print(f"Agent policy: ", policy)
print(f"Agent critic: ", critic)


# Collector setup ======================
collector = SyncDataCollector(
    create_env_fn=lambda: ParallelEnv(10, create_env, mp_start_method="fork"),
    policy=policy,
    device=device,
    storing_device="cpu",
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    reset_at_each_iter=True,
)
