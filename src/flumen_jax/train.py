import sys
from typing import Iterator

import equinox
import jax
import optax
from jax import numpy as jnp
from torch.utils.data import DataLoader

from .dataloader import NumPyLoader
from .model import Flumen
from .typing import BatchedOutput, Inputs


def evaluate(dataloader: NumPyLoader, model: Flumen) -> float:
    total_loss = jnp.array(0.0)
    for y, inputs in dataloader:
        total_loss += equinox.filter_jit(compute_loss)(model, inputs, y)
    return total_loss.item() / len(dataloader)


def torch2jax(dataloader: DataLoader) -> Iterator[tuple[BatchedOutput, Inputs]]:
    for y, x0, rnn_input, tau, lengths in dataloader:
        yield (
            jax.device_put(y.numpy()),
            jax.tree_map(
                jax.device_put,
                (
                    x0.numpy(),
                    rnn_input.numpy(),
                    tau.numpy(),
                    lengths.numpy(),
                ),
            ),
        )


def compute_loss(model: Flumen, inputs: Inputs, y: BatchedOutput) -> jax.Array:
    x, rnn_input, tau, length = inputs
    y_pred = jax.vmap(model)(x, rnn_input, tau, length)
    loss_val = jnp.sum(jnp.square(y - y_pred))

    return loss_val


@equinox.filter_jit
def train_step(
    model: Flumen,
    inputs: Inputs,
    y: BatchedOutput,
    optimiser: optax.GradientTransformation,
    state: optax.OptState,
) -> tuple[Flumen, optax.OptState, jax.Array]:
    loss, grad = equinox.filter_value_and_grad(compute_loss)(model, inputs, y)
    update, new_state = optimiser.update(grad, state)
    model = equinox.apply_updates(model, update)

    return model, new_state, loss


class MetricMonitor:
    patience: int
    atol: float
    rtol: float

    def __init__(self, patience: int, rtol: float, atol: float):
        self._best = float("inf")
        self._counter = 0
        self.patience = patience
        self.rtol = rtol
        self.atol = atol
        self._is_best = False

    def update(self, metric: float) -> bool:
        if self.better(metric):
            self._best = metric
            self._is_best = True
            self._counter = 0
            return False

        self._is_best = False
        self._counter += 1

        if self._counter > self.patience:
            self._counter = 0
            return True

        return False

    def better(self, metric: float) -> bool:
        return metric + self.atol < (1 - self.rtol) * self._best

    @property
    def is_best(self):
        return self._is_best

    @property
    def best_metric(self):
        return self._best


def reduce_learning_rate(state: optax.OptState, factor: float, eps: float):
    curr_lr = state.hyperparams["learning_rate"]  # type:ignore
    new_lr = curr_lr / factor

    if curr_lr - new_lr > eps:
        state.hyperparams["learning_rate"] = new_lr  # type: ignore
        print(
            f"Learning rate reduced to {state.hyperparams['learning_rate']:.2e}",  # type: ignore
            file=sys.stderr,
        )
