import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from dataclasses import dataclass
from typing import Tuple, Optional
import time
import optax

# Try scikit-fda, fallback to simple implementation
try:
    from skfda.representation.basis import BSplineBasis
    from skfda.misc.operators import LinearDifferentialOperator
    from skfda.misc.regularization import L2Regularization

    HAS_SKFDA = True
except ImportError:
    HAS_SKFDA = False


@dataclass
class Spectrogram:
    times: jnp.ndarray  # shape (n_time,)
    freqs: jnp.ndarray  # shape (n_freq,)
    power: jnp.ndarray  # shape (n_time, n_freq)

    @property
    def n_time(self):
        return len(self.times)

    @property
    def n_freq(self):
        return len(self.freqs)


@dataclass
class TimeVaryingLogPSplines:
    time_degree: int
    freq_degree: int
    n_time: int
    n_freq: int
    time_basis: jnp.ndarray
    freq_basis: jnp.ndarray
    kronecker_penalty: jnp.ndarray
    time_knots: np.ndarray
    freq_knots: np.ndarray
    weights: jnp.ndarray

    @property
    def n_time_basis(self) -> int:
        return len(self.time_knots) + self.time_degree - 1

    @property
    def n_freq_basis(self) -> int:
        return len(self.freq_knots) + self.freq_degree - 1

    def __call__(self, weights: jnp.ndarray = None) -> jnp.ndarray:
        if weights is None:
            weights = self.weights
        if weights.ndim == 1:
            weights = weights.reshape(self.n_time_basis, self.n_freq_basis)
        return self.time_basis @ weights @ self.freq_basis.T


def create_kronecker_penalty(time_penalty: jnp.ndarray, freq_penalty: jnp.ndarray) -> jnp.ndarray:
    """Create 2D penalty matrix: P_time ⊗ I + I ⊗ P_freq"""
    n_time, n_freq = time_penalty.shape[0], freq_penalty.shape[0]
    I_time = jnp.eye(n_time)
    I_freq = jnp.eye(n_freq)
    return jnp.kron(time_penalty, I_freq) + jnp.kron(I_time, freq_penalty)


def init_basis_and_penalty_1d(knots: np.ndarray, degree: int, n_grid_points: int,
                              diffMatrixOrder: int, epsilon: float = 1e-6):
    """Generate 1D B-spline basis and penalty matrix."""
    if HAS_SKFDA:
        order = degree + 1
        basis = BSplineBasis(domain_range=[0, 1], order=order, knots=knots)
        grid_points = np.linspace(0, 1, n_grid_points)
        basis_matrix = basis.to_basis().to_grid(grid_points).data_matrix.squeeze().T

        regularization = L2Regularization(LinearDifferentialOperator(diffMatrixOrder))
        penalty = regularization.penalty_matrix(basis)
        penalty = penalty / np.max(penalty) + epsilon * np.eye(penalty.shape[1])

        return jnp.array(basis_matrix), jnp.array(penalty)
    else:
        # Simple fallback
        n_basis = len(knots) + degree - 1
        grid = np.linspace(0, 1, n_grid_points)
        basis_matrix = np.zeros((n_grid_points, n_basis))

        for i in range(n_basis):
            center = i / (n_basis - 1)
            width = 2.0 / n_basis
            basis_matrix[:, i] = np.maximum(0, 1 - np.abs(grid - center) / width)

        penalty = np.eye(n_basis)
        if diffMatrixOrder == 1:
            for i in range(n_basis - 1):
                penalty[i, i] = 2
                penalty[i, i + 1] = penalty[i + 1, i] = -1

        penalty = penalty + epsilon * np.eye(n_basis)
        return jnp.array(basis_matrix), jnp.array(penalty)


def init_2d_weights(log_spectrogram: jnp.ndarray, model: TimeVaryingLogPSplines, num_steps: int = 2000):
    """Initialize weights using optimization."""

    @jax.jit
    def compute_loss(weights_vec: jnp.ndarray) -> float:
        weights = weights_vec.reshape(model.n_time_basis, model.n_freq_basis)
        ln_model = model.time_basis @ weights @ model.freq_basis.T
        return jnp.mean((log_spectrogram - ln_model) ** 2)

    init_weights = 0.01 * jax.random.normal(jax.random.PRNGKey(42),
                                            (model.n_time_basis * model.n_freq_basis,))

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(init_weights)

    def step(i, state):
        weights, opt_state = state
        loss, grads = jax.value_and_grad(compute_loss)(weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (weights, opt_state)

    final_weights, _ = jax.lax.fori_loop(0, num_steps, step, (init_weights, opt_state))
    return final_weights.reshape(model.n_time_basis, model.n_freq_basis)


def bayesian_2d_model_kronecker(
        log_spectrograms: jnp.ndarray,
        time_basis: jnp.ndarray,
        freq_basis: jnp.ndarray,
        kronecker_penalty: jnp.ndarray,
        alpha_phi, beta_phi, alpha_delta, beta_delta,
):
    """2D Bayesian model with Kronecker penalty and Whittle likelihood."""

    n_time_basis = time_basis.shape[1]
    n_freq_basis = freq_basis.shape[1]
    n_total = n_time_basis * n_freq_basis

    # Sample hyperparameters
    delta = numpyro.sample("delta", dist.Gamma(alpha_delta, beta_delta))
    phi = numpyro.sample("phi", dist.Gamma(alpha_phi, delta * beta_phi))

    # Sample vectorized weights
    weights_vec = numpyro.sample("weights", dist.Normal(0, 1).expand([n_total]).to_event(1))

    # Reshape for tensor product
    weights = weights_vec.reshape(n_time_basis, n_freq_basis)

    # Kronecker penalty term
    penalty_quadform = jnp.dot(weights_vec, jnp.dot(kronecker_penalty, weights_vec))
    log_prior = 0.5 * n_total * jnp.log(phi) - 0.5 * phi * penalty_quadform
    numpyro.factor("ln_prior", log_prior)

    # Build surface and Whittle likelihood
    ln_spline_surface = time_basis @ weights @ freq_basis.T
    integrand = ln_spline_surface + jnp.exp(log_spectrograms - ln_spline_surface)
    whittle_loglik = -0.5 * jnp.sum(integrand)
    numpyro.factor("ln_likelihood", whittle_loglik)


def run_2d_mcmc_kronecker(
        spectrogram: Spectrogram,
        n_time_knots: int = 5,
        n_freq_knots: int = 7,
        time_degree: int = 2,
        freq_degree: int = 2,
        num_warmup=500,
        num_samples=300,
        rng_key=0
) -> Tuple[MCMC, TimeVaryingLogPSplines]:
    """Run MCMC with Kronecker penalty structure."""

    # Create knots
    time_knots = np.linspace(0, 1, n_time_knots)
    freq_knots = np.linspace(0, 1, n_freq_knots)

    # Create basis functions and penalties
    time_basis, time_penalty = init_basis_and_penalty_1d(time_knots, time_degree, spectrogram.n_time, 2)
    freq_basis, freq_penalty = init_basis_and_penalty_1d(freq_knots, freq_degree, spectrogram.n_freq, 2)

    # Create Kronecker penalty
    kronecker_penalty = create_kronecker_penalty(time_penalty, freq_penalty)
    eps = 1e-6
    kronecker_penalty = kronecker_penalty + eps * jnp.eye(kronecker_penalty.shape[0])

    # Create model
    model = TimeVaryingLogPSplines(
        time_degree=time_degree, freq_degree=freq_degree,
        n_time=spectrogram.n_time, n_freq=spectrogram.n_freq,
        time_basis=time_basis, freq_basis=freq_basis,
        kronecker_penalty=kronecker_penalty,
        time_knots=time_knots, freq_knots=freq_knots,
        weights=jnp.zeros((len(time_knots) + time_degree - 1, len(freq_knots) + freq_degree - 1))
    )

    # Initialize weights
    print("Optimizing initial weights...")
    model.weights = init_2d_weights(jnp.log(spectrogram.power), model)

    print(f"Model: {model.n_time_basis} × {model.n_freq_basis} = {model.n_time_basis * model.n_freq_basis} parameters")

    # PLOT INITIAL FIT BEFORE MCMC
    print("Plotting initial optimization fit...")
    plot_initial_fit(spectrogram, model)

    # MCMC
    init_strategy = init_to_value(values=dict(
        delta=1e-3, phi=1.0,
        weights=model.weights.reshape(-1)
    ))

    kernel = NUTS(bayesian_2d_model_kronecker, init_strategy=init_strategy, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)

    t0 = time.time()
    mcmc.run(
        jax.random.PRNGKey(rng_key),
        jnp.log(spectrogram.power),
        model.time_basis,
        model.freq_basis,
        model.kronecker_penalty,
        1.0, 1.0, 1e-4, 1e-4  # hyperparameters
    )

    print(f"MCMC completed in {time.time() - t0:.2f}s")

    # Check divergences
    divergences = mcmc.get_extra_fields()['diverging']
    print(f"Divergences: {jnp.sum(divergences)}/{len(divergences)}")

    return mcmc, model


def plot_initial_fit(spectrogram: Spectrogram, model: TimeVaryingLogPSplines):
    """Plot initial optimization fit before MCMC."""

    # Compute initial fit using optimized weights
    initial_log_surface = model.time_basis @ model.weights @ model.freq_basis.T
    initial_surface = jnp.exp(initial_log_surface)

    # Compute residuals
    log_residuals = jnp.log(spectrogram.power) - initial_log_surface
    power_residuals = spectrogram.power - initial_surface

    # Compute fit metrics
    log_mse = jnp.mean(log_residuals ** 2)
    power_mse = jnp.mean(power_residuals ** 2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Top row: log scale
    # Original log data
    im1 = axes[0, 0].pcolormesh(spectrogram.freqs, spectrogram.times,
                                jnp.log(spectrogram.power), shading='auto')
    axes[0, 0].set_title('Log Data')
    axes[0, 0].set_ylabel('Time')
    plt.colorbar(im1, ax=axes[0, 0])

    # Initial log fit
    im2 = axes[0, 1].pcolormesh(spectrogram.freqs, spectrogram.times,
                                initial_log_surface, shading='auto')
    axes[0, 1].set_title(f'Initial Log Fit (MSE={log_mse:.4f})')
    plt.colorbar(im2, ax=axes[0, 1])

    # Log residuals
    im3 = axes[0, 2].pcolormesh(spectrogram.freqs, spectrogram.times,
                                log_residuals, shading='auto', cmap='RdBu_r')
    axes[0, 2].set_title('Log Residuals')
    plt.colorbar(im3, ax=axes[0, 2])

    # Bottom row: power scale
    # Original power data
    im4 = axes[1, 0].pcolormesh(spectrogram.freqs, spectrogram.times,
                                spectrogram.power, shading='auto')
    axes[1, 0].set_title('Power Data')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Time')
    plt.colorbar(im4, ax=axes[1, 0])

    # Initial power fit
    im5 = axes[1, 1].pcolormesh(spectrogram.freqs, spectrogram.times,
                                initial_surface, shading='auto')
    axes[1, 1].set_title(f'Initial Power Fit (MSE={power_mse:.4f})')
    axes[1, 1].set_xlabel('Frequency')
    plt.colorbar(im5, ax=axes[1, 1])

    # Power residuals
    im6 = axes[1, 2].pcolormesh(spectrogram.freqs, spectrogram.times,
                                power_residuals, shading='auto', cmap='RdBu_r')
    axes[1, 2].set_title('Power Residuals')
    axes[1, 2].set_xlabel('Frequency')
    plt.colorbar(im6, ax=axes[1, 2])

    plt.suptitle('Initial Optimization Fit (Before MCMC)', fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Initial fit metrics:")
    print(f"  Log MSE: {log_mse:.6f}")
    print(f"  Power MSE: {power_mse:.6f}")
    print(f"  Max log residual: {jnp.max(jnp.abs(log_residuals)):.4f}")
    print(f"  Mean absolute log residual: {jnp.mean(jnp.abs(log_residuals)):.4f}")

    return initial_surface, log_residuals


def plot_results(spectrogram: Spectrogram, model: TimeVaryingLogPSplines, mcmc: MCMC):
    """Plot 2D results."""
    samples = mcmc.get_samples()
    mean_weights = jnp.mean(samples['weights'], axis=0).reshape(model.n_time_basis, model.n_freq_basis)
    fitted_surface = jnp.exp(model.time_basis @ mean_weights @ model.freq_basis.T)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Data
    im1 = axes[0].pcolormesh(spectrogram.freqs, spectrogram.times, spectrogram.power, shading='auto')
    axes[0].set_title('Data')
    plt.colorbar(im1, ax=axes[0])

    # Fit
    im2 = axes[1].pcolormesh(spectrogram.freqs, spectrogram.times, fitted_surface, shading='auto')
    axes[1].set_title('Fitted')
    plt.colorbar(im2, ax=axes[1])

    # Residuals
    im3 = axes[2].pcolormesh(spectrogram.freqs, spectrogram.times,
                             spectrogram.power - fitted_surface, shading='auto', cmap='RdBu_r')
    axes[2].set_title('Residuals')
    plt.colorbar(im3, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Time')

    plt.tight_layout()
    return fig


def generate_test_data(n_time=25, n_freq=40):
    """Generate synthetic time-varying spectral data."""
    times = np.linspace(0, 10, n_time)
    freqs = np.linspace(0.1, 1.0, n_freq)
    T, F = np.meshgrid(times, freqs, indexing='ij')

    # Moving spectral peak + background
    peak_freq = 0.3 + 0.2 * np.sin(2 * np.pi * T / 8)
    true_psd = 1.0 / (1 + F ** 2) + 2.0 * np.exp(-((F - peak_freq) / 0.05) ** 2)

    # Add noise
    observed_power = true_psd * (1 + 0.15 * np.random.randn(*true_psd.shape))
    observed_power = np.maximum(observed_power, 0.01)

    return Spectrogram(times=jnp.array(times), freqs=jnp.array(freqs), power=jnp.array(observed_power))


def main():
    """Main example with Kronecker penalty."""
    print("Generating test data...")
    spectrogram = generate_test_data()

    print("Running 2D P-spline MCMC with Kronecker penalty...")
    mcmc, model = run_2d_mcmc_kronecker(
        spectrogram,
        n_time_knots=15,
        n_freq_knots=15,
        num_warmup=400,
        num_samples=200
    )

    print("Plotting results...")
    fig = plot_results(spectrogram, model, mcmc)
    plt.show()

    return mcmc, model


if __name__ == "__main__":
    mcmc, model = main()