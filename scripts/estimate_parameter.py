import equinox
import yaml
from jax import random as jrd
from argparse import ArgumentParser
from pathlib import Path
from flumen_jax import Flumen
from flumen_jax.parameter_estimation import ParameterEstimator
from semble import make_trajectory_sampler, TSamplerSpec, TrajectorySampler


def parse_args():
    ap = ArgumentParser()
    ap.add_argument(
        "path",
        type=str,
        help="Path to .pth file "
        "(or, if run with --wandb, path to a Weights & Biases artifact)",
    )
    ap.add_argument("--wandb", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    # load in metadata
    if args.wandb:
        import wandb

        api = wandb.Api()
        model_artifact = api.artifact(args.path)
        model_path = Path(model_artifact.download())

    else:
        model_path = Path(args.path)

    with open(model_path / "metadata.yaml", "r") as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    # load in model and sampler
    model: Flumen = equinox.filter_eval_shape(
        Flumen, **metadata["args"], key=jrd.key(0)
    )
    model: Flumen = equinox.tree_deserialise_leaves(
        model_path / "leaves.eqx", model
    )

    sampler_spec: TSamplerSpec = metadata["data_settings"]
    sampler: TrajectorySampler = make_trajectory_sampler(sampler_spec)
    _, _, output_dim = sampler.dims()
    sampler.reset_rngs()

    # load in parameter estimator
    parameter_estimator = ParameterEstimator(model, sampler)
    parameter_estimator.reset_rngs()
    _ = parameter_estimator()  # predict parameter


if __name__ == "__main__":
    main()
