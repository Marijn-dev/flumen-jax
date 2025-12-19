import sys
from typing import Iterator

import equinox
import jax
import optax
from jax import numpy as jnp
from torch.utils.data import DataLoader
from jaxtyping import PyTree, PyTreeDef

from .dataloader import NumPyLoader
from .model import Flumen
from .typing import BatchedOutput, Inputs, Inputs_withParam


def evaluate(
    dataloader: NumPyLoader, flat_model: PyTree, model_treedef: PyTreeDef
) -> float:
    total_loss = jnp.array(0.0)
    for y, inputs in dataloader:
        total_loss += compute_loss_flat(flat_model, model_treedef, inputs, y)
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


@equinox.filter_jit
def compute_loss_flat(
    flat_model,
    model_treedef: PyTreeDef,
    inputs: Inputs | Inputs_withParam,
    y: BatchedOutput,
) -> jax.Array:
    model = jax.tree_util.tree_unflatten(model_treedef, flat_model)
    x, rnn_input, tau, length, *parameter = inputs
    parameter = parameter[0] if parameter else None
    if parameter is None:
        y_pred = jax.vmap(model)(x, rnn_input, tau, length)
    else:
        y_pred = jax.vmap(model)(x, rnn_input, tau, length, parameter)
    loss_val = jnp.sum(jnp.square(y - y_pred))

    return loss_val


def compute_loss(
    model: Flumen, inputs: Inputs | Inputs_withParam, y: BatchedOutput
) -> jax.Array:
    x, rnn_input, tau, length, *parameter = inputs
    parameter = parameter[0] if parameter else None
    if parameter is None:
        y_pred = jax.vmap(model)(x, rnn_input, tau, length)
    else:
        y_pred = jax.vmap(model)(x, rnn_input, tau, length, parameter)
    loss_val = jnp.sum(jnp.square(y - y_pred))

    return loss_val


@equinox.filter_jit
def train_step(
    flat_model: PyTree,
    model_treedef: PyTreeDef,
    inputs: Inputs | Inputs_withParam,
    y: BatchedOutput,
    optimiser: optax.GradientTransformation,
    flat_state: PyTree,
    state_treedef: PyTreeDef,
) -> tuple[PyTree, PyTree, jax.Array]:
    model = jax.tree_util.tree_unflatten(model_treedef, flat_model)
    state = jax.tree_util.tree_unflatten(state_treedef, flat_state)
    loss, grad = equinox.filter_value_and_grad(compute_loss)(model, inputs, y)
    update, new_state = optimiser.update(grad, state)
    model = equinox.apply_updates(model, update)

    flat_model = jax.tree_util.tree_leaves(model)
    flat_state = jax.tree_util.tree_leaves(new_state)

    return flat_model, flat_state, loss


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
