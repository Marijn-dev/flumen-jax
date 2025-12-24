import jax
import optimistix as optx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Float
from time import time
from flumen_jax import Flumen, Parameter
from semble import TrajectorySampler


class ParameterEstimator:
    def __init__(
        self,
        model: Flumen,
        sampler: TrajectorySampler,
    ):
        self.model = model
        self.sampler = sampler
        self.solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)

        self._rng = np.random.default_rng(seed=42)
        self._true_param_rng, self._init_param_rng = self._rng.spawn(2)

    def _create_data(
        self,
        nr_trajectories: int,
        parameter: Parameter,
        time_horizon: Float = 10.0,
    ) -> None:
        x0_data = []
        y_data = []
        t_data = []
        u_data = []
        for _ in range(nr_trajectories):
            x0, t, y, u, _ = self.sampler.get_example(
                time_horizon=time_horizon,
                n_samples=int(1 + 20 * time_horizon),
                parameter=parameter,
            )
            x0_data.append(x0)
            y_data.append(y)
            t_data.append(t)
            u_data.append(u)

        self.x0_data = np.array(x0_data)
        self.y_data = np.array(y_data)
        self.t_data = np.array(t_data)
        self.u_data = np.array(u_data)

    def reset_rngs(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed=seed)
        self._true_param_rng, self._init_param_rng = self._rng.spawn(2)

    def _eval_trajectory(self, model: Flumen, t, x0, u, delta, parameter):
        skips = jnp.floor(t / delta).astype(jnp.uint32)
        tau = (t - delta * skips) / delta
        eval_trajectory_vmapped = jax.vmap(
            model.eval_trajectory, in_axes=(0, 0, 0, 0, None)
        )
        eval_trajectory_vmapped_jit = jax.jit(eval_trajectory_vmapped)
        y_pred = eval_trajectory_vmapped_jit(
            x0, u, tau, skips.squeeze(), parameter
        )
        return y_pred

    def _residuals(self, parameters, values):
        y, model, t, x0, u, delta = values
        y_pred = self._eval_trajectory(model, t, x0, u, delta, parameters)
        return y - y_pred

    def __call__(
        self,
        true_parameter: Parameter | None = None,
        init_parameter: Parameter | None = None,
    ) -> Parameter:
        true_parameter = (
            true_parameter
            if true_parameter is not None
            else self.sampler._dyn._parameter_generator.sample(
                self._true_param_rng
            )
        )
        init_parameter = (
            init_parameter
            if init_parameter is not None
            else self.sampler._dyn._parameter_generator.sample(
                self._init_param_rng
            )
        )

        # create data
        self._create_data(
            nr_trajectories=10, parameter=true_parameter, time_horizon=10
        )

        # mock call for jit compilation
        _ = optx.least_squares(
            self._residuals,
            self.solver,
            init_parameter,
            args=(
                self.y_data,
                self.model,
                self.t_data,
                self.x0_data,
                self.u_data,
                self.sampler._delta,
            ),
        )  # parameter is entered by solver

        # find parameter
        time_predict = time()
        problem = optx.least_squares(
            self._residuals,
            self.solver,
            init_parameter,
            args=(
                self.y_data,
                self.model,
                self.t_data,
                self.x0_data,
                self.u_data,
                self.sampler._delta,
            ),
        )  # parameter is entered by solver
        time_predict = time() - time_predict
        est_parameter = problem.value
        print(
            "Optimization time:",
            time_predict,
            " True parameter:",
            true_parameter,
            "Initial parameter:",
            init_parameter,
            "Estimated Parameter:",
            est_parameter,
        )
        return est_parameter
