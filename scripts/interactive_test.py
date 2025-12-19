from jax import random as jrd
from jax import numpy as jnp

import equinox

import matplotlib.pyplot as plt
import numpy as np

from flumen_jax import Flumen
from semble import make_trajectory_sampler, TSamplerSpec
from semble.dynamics import ContinuousStateDynamics

from argparse import ArgumentParser

import yaml
from pathlib import Path
import sys
from pprint import pprint
from time import time
from typing import cast


@equinox.filter_jit
def eval_trajectory(model: Flumen, t, x0, u, delta, parameter=None):
    skips = jnp.floor(t / delta).astype(jnp.uint32)
    tau = (t - delta * skips) / delta

    return model.eval_trajectory(x0, u, tau, skips.squeeze(), parameter)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument(
        "path",
        type=str,
        help="Path to .pth file "
        "(or, if run with --wandb, path to a Weights & Biases artifact)",
    )
    ap.add_argument(
        "--print_info",
        action="store_true",
        help="Print training metadata and quit",
    )
    ap.add_argument("--continuous_state", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--time_horizon", type=float, default=None)

    return ap.parse_args()


def main():
    args = parse_args()

    if args.wandb:
        import wandb

        api = wandb.Api()
        model_artifact = api.artifact(args.path)
        model_path = Path(model_artifact.download())

        model_run = model_artifact.logged_by()
        if model_run:
            pprint(model_run.summary)
    else:
        model_path = Path(args.path)

    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    pprint(metadata)

    if args.print_info:
        return
    model: Flumen = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )
    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", model
    )

    sampler_spec: TSamplerSpec = metadata["data_settings"]
    sampler = make_trajectory_sampler(sampler_spec)
    _, _, output_dim = sampler.dims()
    sampler.reset_rngs()
    delta = sampler._delta

    if args.continuous_state:
        dynamics = cast(ContinuousStateDynamics, sampler._dyn)
        xx = dynamics.get_space_axis()
        n_plots = 2
    else:
        n_plots = output_dim
        xx = None

    fig, ax = plt.subplots(n_plots + 1, 1, sharex=True)
    fig.canvas.mpl_connect("close_event", on_close_window)

    time_horizon = (
        args.time_horizon
        if args.time_horizon
        else metadata["data_args"]["time_horizon"]
    )

    while True:
        time_integrate = time()
        if sampler._dyn._is_parameterised:
            x0, t, y, u, parameter = sampler.get_example(
                time_horizon=time_horizon, n_samples=int(1 + 20 * time_horizon)
            )
            print("parameter(s) this trajectory: ", parameter)
        else:
            x0, t, y, u = sampler.get_example(
                time_horizon=time_horizon, n_samples=int(1 + 20 * time_horizon)
            )
            parameter = None

        time_integrate = time() - time_integrate

        time_predict = time()
        y_pred = eval_trajectory(model, t, x0, u, delta, parameter)
        time_predict = time() - time_predict

        print(f"Timings: {time_integrate}, {time_predict}")

        y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]

        sq_error = np.square(y - y_pred)
        print(output_dim * np.mean(sq_error))

        if args.continuous_state:
            ax[0].pcolormesh(t.squeeze(), xx, y.T)
            ax[1].pcolormesh(t.squeeze(), xx, y_pred.T)
        else:
            for k, ax_ in enumerate(ax[:output_dim]):
                ax_.plot(t, y_pred[:, k], c="orange", label="Model output")
                ax_.plot(t, y[:, k], "b--", label="True state")
                ax_.set_ylabel(f"$x_{k + 1}$")

        ax[-1].step(np.arange(0.0, time_horizon, delta), u[:-1], where="post")
        ax[-1].set_ylabel("$u$")
        ax[-1].set_xlabel("$t$")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        plt.draw()

        # Wait for key press
        skip = False
        while not skip:
            skip = plt.waitforbuttonpress()

        for ax_ in ax:
            ax_.clear()


def on_close_window(_):
    sys.exit(0)


if __name__ == "__main__":
    main()
