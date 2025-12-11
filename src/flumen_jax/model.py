import jax
from jax import numpy as jnp
from jax import random as jrd

from jaxtyping import Float, UInt, Array, PRNGKeyArray

import equinox
from equinox.nn import LSTMCell, MLP

State = Float[Array, "state_dim"]
BatchedState = Float[Array, "batch state_dim"]

Output = Float[Array, "output_dim"]
BatchedOutput = Float[Array, "batch output_dim"]

Input = Float[Array, "seq_len control_dim"]
RNNInput = Float[Array, "seq_len control_dim+1"]
BatchedRNNInput = Float[Array, "batch seq_len control_dim+1"]

RNNState = Float[Array, "feature_dim"]

TimeIncrement = Float[Array, "1"]
BatchedTimeIncrement = Float[Array, "batch 1"]

BatchLengths = UInt[Array, "batch 1"]


class FlumenHead(equinox.Module):
    cell: LSTMCell
    encoder: MLP

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        feature_dim: int,
        encoder_hsz: int,
        key: PRNGKeyArray,
    ):
        lstm_key, enc_key = jrd.split(key, 2)

        self.cell = LSTMCell(
            input_size=control_dim + 1, hidden_size=feature_dim, key=lstm_key
        )

        self.encoder = MLP(
            in_size=state_dim,
            out_size=feature_dim,
            width_size=encoder_hsz,
            depth=2,
            activation=jnp.tanh,
            key=enc_key,
        )

    def __call__(
        self, initial_state: State, rnn_input: RNNInput
    ) -> tuple[RNNState, Array]:
        h = self.encoder(initial_state)
        c = jnp.zeros_like(h)

        (h_last, _), h_seq = jax.lax.scan(
            lambda state, input: (self.cell(input, state), state[0]),
            (h, c),
            rnn_input,
        )

        return h_last, h_seq

    def eval_trajectory(
        self,
        initial_state: State,
        u: Input,
        tau: Float[Array, "n_time_pts 1"],  # noqa: F722
        skips: UInt[Array, "n_time_pts"],  # noqa: F821
    ) -> tuple[RNNState, RNNState]:
        h = self.encoder(initial_state)
        c = jnp.zeros_like(h)

        rnn_input_integer_steps = jnp.concatenate(
            (u, jnp.ones((u.shape[0], 1))), axis=-1
        )

        (_, _), rnn_state_seq = jax.lax.scan(
            lambda state, input: (self.cell(input, state), state),
            (h, c),
            rnn_input_integer_steps,
        )

        rnn_input_tau = jnp.concatenate((u[skips], tau), axis=-1)
        h0 = rnn_state_seq[0][skips, :]
        c0 = rnn_state_seq[1][skips, :]

        h1, _ = jax.vmap(self.cell)(rnn_input_tau, (h0, c0))

        return h0, h1


class FlumenTail(equinox.Module):
    decoder: MLP

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        decoder_hsz: int,
        key: PRNGKeyArray,
    ):
        self.decoder = MLP(
            in_size=feature_dim,
            out_size=output_dim,
            width_size=decoder_hsz,
            depth=2,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(
        self, h0: RNNState, h1: RNNState, tau: TimeIncrement
    ) -> Output:
        return self.decoder((1 - tau) * h0 + tau * h1)


class Flumen(equinox.Module):
    head: FlumenHead
    tail: FlumenTail

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        output_dim: int,
        feature_dim: int,
        encoder_hsz: int,
        decoder_hsz: int,
        key: PRNGKeyArray,
    ):
        hkey, tkey = jrd.split(key, 2)

        self.head = FlumenHead(
            state_dim, control_dim, feature_dim, encoder_hsz, hkey
        )

        self.tail = FlumenTail(feature_dim, output_dim, decoder_hsz, tkey)

    def __call__(
        self,
        initial_state: BatchedState,
        rnn_input: BatchedRNNInput,
        tau: BatchedTimeIncrement,
        batch_lens: BatchLengths,
    ) -> BatchedOutput:
        bs = initial_state.shape[0]
        max_seq_len = rnn_input.shape[1]

        h_last, h_seq = jax.vmap(self.head)(initial_state, rnn_input)
        h0 = h_seq[jnp.arange(bs), batch_lens - 1]
        h1 = jnp.where(
            jnp.expand_dims(batch_lens, -1) < max_seq_len,
            h_seq[jnp.arange(bs), batch_lens],
            h_last,
        )

        y = jax.vmap(self.tail)(h0, h1, tau)

        return y

    def eval_trajectory(
        self,
        initial_state: State,
        u: Input,
        tau: Float[Array, "n_time_pts 1"],  # noqa: F722
        skips: UInt[Array, "n_time_pts"],  # noqa: F821
    ):
        h0, h1 = self.head.eval_trajectory(initial_state, u, tau, skips)

        y = jax.vmap(self.tail)(h0, h1, tau)

        return y
