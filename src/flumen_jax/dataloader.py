from typing import Iterator

import jax
import numpy as np
from flumen import TrajectoryDataset, ParameterisedTrajectoryDataset
from jaxtyping import Array, Float, UInt

from .typing import BatchedOutput, Inputs


class NumPyDataset:
    y: Float[Array, "dlen output_dim"]
    initial_state: Float[Array, "dlen state_dim"]
    rnn_input: Float[Array, "dlen seq_len "]
    tau: Float[Array, "dlen 1"]
    lengths: UInt[Array, "dlen 1"]

    state_dim: int
    output_dim: int
    control_dim: int

    def __init__(self, data: TrajectoryDataset):
        (
            self.y,
            self.initial_state,
            self.rnn_input,
            self.tau,
            lengths,
        ) = jax.tree_map(
            np.asarray,
            (
                data.state,
                data.init_state,
                data.rnn_input,
                data.tau,
                data.seq_lens,
            ),
        )
        self.lengths = lengths.astype(np.uint32)

        self.state_dim = data.state_dim
        self.output_dim = data.output_dim
        self.control_dim = data.control_dim
        self.parameter_dim = data.parameter_dim

    def __getitem__(self, index) -> tuple[BatchedOutput, Inputs]:
        return (
            self.y[index],
            (
                self.initial_state[index],
                self.rnn_input[index],
                self.tau[index],
                self.lengths[index],
            ),
        )

    def __len__(self):
        return self.y.shape[0]


class ParameterisedNumPyDataset(NumPyDataset):
    parameter: Float[Array, "dlen parameter_dim"]

    def __init__(self, data: ParameterisedTrajectoryDataset):
        super().__init__(data)

        self.parameter = jax.tree_map(np.asarray, data.parameter)

    def __getitem__(self, index):
        y, inputs = super().__getitem__(index)
        return y, (*inputs, self.parameter[index])


def dataloader(
    data: NumPyDataset | ParameterisedNumPyDataset,
    batch_size,
    shuffle,
    skip_last,
) -> Iterator[tuple[BatchedOutput, Inputs]]:
    dlen = len(data)
    indices = np.arange(dlen)
    if shuffle:
        indices = np.random.permutation(indices)
    start = 0
    end = batch_size
    while end <= dlen:
        batch_perm = indices[start:end]
        yield data[batch_perm]
        start = end
        end = start + batch_size
    if not skip_last and start < dlen:
        yield data[start:dlen]


class NumPyLoader:
    def __init__(
        self,
        data: NumPyDataset | ParameterisedNumPyDataset,
        batch_size: int,
        shuffle=True,
        skip_last=True,
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.skip_last = skip_last

    def __iter__(self):
        return dataloader(
            self.data, self.batch_size, self.shuffle, self.skip_last
        )

    def __len__(self):
        if self.skip_last:
            return len(self.data) - len(self.data) % self.batch_size

        return len(self.data)
