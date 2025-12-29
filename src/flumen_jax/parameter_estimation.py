import optimistix as optx
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
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
        # self.solver = optx.LevenbergMarquardt(
        # rtol=1e-8, atol=1e-8, verbose=frozenset({"step", "accepted", "loss", "step_size"})) # use this if you want additional print statements

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

    def _get_trajectory(self, t_samples, x0, u, parameter):
        """Generate trajectories given inputs (x0,t,u, parameter) using Diffrax"""
        t_samples = t_samples.reshape(-1)
        self.initial_time, self.end_time = t_samples[0], t_samples[-1]

        def f(t, x, args):
            u, parameter = args
            n_control = jnp.array(
                (t - self.initial_time) / self.sampler._delta, dtype=int
            )
            u_val = u[n_control]
            self.sampler._dyn.gen_parameter(
                None, parameter
            )  # set parameter in sampler dynamics
            return jnp.stack([*self.sampler._dyn(x, u_val)])

        term = diffrax.ODETerm(f)
        solver = diffrax.Dopri5()

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=self.initial_time,
            t1=self.end_time,
            dt0=0.1,
            y0=x0,
            args=(u, parameter),
            saveat=diffrax.SaveAt(ts=t_samples),
            adjoint=diffrax.DirectAdjoint(),
        )
        return solution.ys

    def _residuals_flow(self, parameter, values):
        """Calculate residual between flow model with estimated parameter and data with true parameter"""
        y, model, t, x0, u, delta = values
        skips = jnp.floor(t / delta).astype(jnp.uint32)
        tau = (t - delta * skips) / delta
        eval_trajectory_vmapped = eqx.filter_vmap(
            model.eval_trajectory, in_axes=(0, 0, 0, 0, None)
        )
        eval_trajectory_vmapped_jit = eqx.filter_jit(eval_trajectory_vmapped)
        y_pred = eval_trajectory_vmapped_jit(
            x0, u, tau, skips.squeeze(), parameter
        )
        return y - y_pred

    def _residuals_ode(self, parameter, values):
        """Calculate residual between the ODE with estimated parameter and data with true parameter"""
        y, _, t, x0, u, _ = values
        eval_trajectory_vmapped = eqx.filter_vmap(
            self._get_trajectory, in_axes=(0, 0, 0, None)
        )
        eval_trajectory_vmapped_jit = eqx.filter_jit(eval_trajectory_vmapped)
        y_pred = eval_trajectory_vmapped_jit(t, x0, u, parameter)
        return y - y_pred

    def __call__(
        self,
        true_parameter: Parameter | None = None,
        init_parameter: Parameter | None = None,
    ) -> tuple[Parameter, Parameter]:
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

        def print_statement(est_time, true_param, init_param, est_param):
            print(
                f"Optimization time: {est_time:.3f}, True parameter: {true_param:.3f}, Initial parameter: {init_param:.3f}, Estimated parameter: {est_param:.3f}"
            )

        # create data
        self._create_data(
            nr_trajectories=10, parameter=true_parameter, time_horizon=10
        )

        print("Solving using ODE...")
        # mock / first call for jit compilation
        time_predict = time()
        problem = optx.least_squares(
            self._residuals_ode,
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
        est_parameter_ode = problem.value
        time_predict = time() - time_predict
        print_statement(
            time_predict,
            true_parameter.item(),
            init_parameter.item(),
            est_parameter_ode.item(),
        )
        # precompiled / second call
        time_predict = time()
        problem = optx.least_squares(
            self._residuals_ode,
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
        est_parameter_ode = problem.value
        time_predict = time() - time_predict
        print_statement(
            time_predict,
            true_parameter.item(),
            init_parameter.item(),
            est_parameter_ode.item(),
        )

        print("Solving using model...")
        # mock / first call for jit compilation
        time_predict = time()
        problem = optx.least_squares(
            self._residuals_flow,
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
        est_parameter = problem.value
        time_predict = time() - time_predict
        print_statement(
            time_predict,
            true_parameter.item(),
            init_parameter.item(),
            est_parameter.item(),
        )

        # precompiled / second call
        time_predict = time()
        problem = optx.least_squares(
            self._residuals_flow,
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
        est_parameter_flow = problem.value
        time_predict = time() - time_predict
        print_statement(
            time_predict,
            true_parameter.item(),
            init_parameter.item(),
            est_parameter_flow.item(),
        )

        return (est_parameter_ode, est_parameter_flow)
