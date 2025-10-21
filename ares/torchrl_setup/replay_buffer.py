from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer
from ares.torchrl_setup.hyperparameters_and_setup import (
    frames_per_batch,
    minibatch_size,
)


sampler = SamplerWithoutReplacement()
storage = LazyTensorStorage(frames_per_batch, device="cpu")
buffer = ReplayBuffer(
    storage=storage,  # We store the frames_per_batch collected at each iteration
    sampler=sampler,
    batch_size=minibatch_size,  # We will sample minibatches of this size
)
