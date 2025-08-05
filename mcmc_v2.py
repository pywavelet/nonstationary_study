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


def plot_initial_fit(spectrogram: Spectrogram, model: TimeVaryingLogPSplines, save_path="initial_fit.png"):
    """Plot and save initial optimization fit before MCMC."""

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
    im1 = axes[0, 0].pcolormesh(spectrogram.freqs, spectrogram.times,
                                jnp.log(spectrogram.power), shading='auto')
    axes[0, 0].set_title('Log Data')
    axes[0, 0].set_ylabel('Time')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(spectrogram.freqs, spectrogram.times,
                                initial_log_surface, shading='auto')
    axes[0, 1].set_title(f'Initial Log Fit (MSE={log_mse:.4f})')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].pcolormesh(spectrogram.freqs, spectrogram.times,
                                log_residuals, shading='auto', cmap='RdBu_r')
    axes[0, 2].set_title('Log Residuals')
    plt.colorbar(im3, ax=axes[0, 2])

    # Bottom row: power scale
    im4 = axes[1, 0].pcolormesh(spectrogram.freqs, spectrogram.times,
                                spectrogram.power, shading='auto')
    axes[1, 0].set_title('Power Data')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Time')
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].pcolormesh(spectrogram.freqs, spectrogram.times,
                                initial_surface, shading='auto')
    axes[1, 1].set_title(f'Initial Power Fit (MSE={power_mse:.4f})')
    axes[1, 1].set_xlabel('Frequency')
    plt.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].pcolormesh(spectrogram.freqs, spectrogram.times,
                                power_residuals, shading='auto', cmap='RdBu_r')
    axes[1, 2].set_title('Power Residuals')
    axes[1, 2].set_xlabel('Frequency')
    plt.colorbar(im6, ax=axes[1, 2])

    plt.suptitle('Initial Optimization Fit (Before MCMC)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Initial fit saved to {save_path}")
    print(f"  Log MSE: {log_mse:.6f}")
    print(f"  Power MSE: {power_mse:.6f}")
    print(f"  Max log residual: {jnp.max(jnp.abs(log_residuals)):.4f}")

    return initial_surface, log_residuals


def plot_penalty_diagnostics(model: TimeVaryingLogPSplines, save_path="penalty_diagnostics.png"):
    """Analyze and plot penalty matrix properties."""

    kronecker_penalty = model.kronecker_penalty
    time_penalty = model.time_penalty_matrix
    freq_penalty = model.freq_penalty_matrix

    # Compute diagnostics
    cond_num = jnp.linalg.cond(kronecker_penalty)
    rank = jnp.linalg.matrix_rank(kronecker_penalty)
    eigenvals = jnp.linalg.eigvals(kronecker_penalty)
    eigenvals = jnp.sort(eigenvals)[::-1]  # Descending order

    # Effective rank (number of "large" eigenvalues)
    total_variance = jnp.sum(eigenvals)
    cumsum_eigs = jnp.cumsum(eigenvals)
    effective_rank = jnp.sum(cumsum_eigs / total_variance < 0.95) + 1

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Individual penalty matrices
    im1 = axes[0, 0].imshow(time_penalty, cmap='RdBu_r')
    axes[0, 0].set_title(f'Time Penalty\n(Cond: {jnp.linalg.cond(time_penalty):.1e})')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(freq_penalty, cmap='RdBu_r')
    axes[0, 1].set_title(f'Freq Penalty\n(Cond: {jnp.linalg.cond(freq_penalty):.1e})')
    plt.colorbar(im2, ax=axes[0, 1])

    # Kronecker penalty (subsampled if too large)
    kronecker_display = kronecker_penalty
    if kronecker_penalty.shape[0] > 100:
        # Subsample for visualization
        idx = jnp.linspace(0, kronecker_penalty.shape[0] - 1, 100, dtype=int)
        kronecker_display = kronecker_penalty[jnp.ix_(idx, idx)]

    im3 = axes[0, 2].imshow(kronecker_display, cmap='RdBu_r')
    axes[0, 2].set_title(f'Kronecker Penalty\n(Cond: {cond_num:.1e}, Rank: {rank})')
    plt.colorbar(im3, ax=axes[0, 2])

    # Bottom row: Eigenvalue analysis
    axes[1, 0].semilogy(eigenvals)
    axes[1, 0].set_title(f'Eigenvalues\n(Effective rank: {effective_rank}/{len(eigenvals)})')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Eigenvalue')
    axes[1, 0].grid(True)

    # Eigenvalue histogram
    axes[1, 1].hist(jnp.log10(eigenvals + 1e-12), bins=30, alpha=0.7)
    axes[1, 1].set_title('Log10 Eigenvalue Distribution')
    axes[1, 1].set_xlabel('log10(eigenvalue)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True)

    # Cumulative eigenvalue variance
    axes[1, 2].plot(cumsum_eigs / total_variance)
    axes[1, 2].axhline(0.95, color='r', linestyle='--', label='95% variance')
    axes[1, 2].axvline(effective_rank, color='r', linestyle='--', label=f'Eff. rank: {effective_rank}')
    axes[1, 2].set_title('Cumulative Eigenvalue Variance')
    axes[1, 2].set_xlabel('Eigenvalue Index')
    axes[1, 2].set_ylabel('Cumulative Variance Fraction')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.suptitle('Penalty Matrix Diagnostics', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Penalty diagnostics saved to {save_path}")
    print(f"  Condition number: {cond_num:.2e}")
    print(f"  Matrix rank: {rank} / {kronecker_penalty.shape[0]}")
    print(f"  Effective rank (95% var): {effective_rank}")
    print(f"  Eigenvalue range: [{jnp.min(eigenvals):.2e}, {jnp.max(eigenvals):.2e}]")

    if cond_num > 1e12:
        print("  ⚠️  WARNING: Very high condition number - may cause numerical issues")
    if effective_rank < kronecker_penalty.shape[0] * 0.1:
        print("  ⚠️  WARNING: Very low effective rank - penalty may be too strong")

    return eigenvals, effective_rank


def plot_mcmc_diagnostics(mcmc: MCMC, model: TimeVaryingLogPSplines, save_prefix="mcmc"):
    """Plot comprehensive MCMC diagnostics and save all plots."""

    samples = mcmc.get_samples()

    # 1. Arviz trace plot
    inf_obj = az.from_numpyro(mcmc)
    fig_trace = az.plot_trace(inf_obj, var_names=['phi', 'delta'])
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_trace.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trace plot saved to {save_prefix}_trace.png")

    # 2. Convergence diagnostics
    summary = az.summary(inf_obj, var_names=['phi', 'delta'])
    print("\nMCMC Summary:")
    print(summary)

    # 3. Energy plot (for diagnosing divergences)
    try:
        fig_energy = az.plot_energy(inf_obj)
        plt.savefig(f"{save_prefix}_energy.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Energy plot saved to {save_prefix}_energy.png")
    except:
        print("Could not create energy plot")

    # 4. Weight parameter diagnostics
    weights_samples = samples['weights']
    n_time_basis = model.n_time_basis
    n_freq_basis = model.n_freq_basis

    # Reshape weight samples
    weights_2d_samples = weights_samples.reshape(-1, n_time_basis, n_freq_basis)

    # Plot weight statistics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Weight means
    weight_means = jnp.mean(weights_2d_samples, axis=0)
    im1 = axes[0, 0].imshow(weight_means, cmap='RdBu_r')
    axes[0, 0].set_title('Posterior Mean Weights')
    plt.colorbar(im1, ax=axes[0, 0])

    # Weight standard deviations
    weight_stds = jnp.std(weights_2d_samples, axis=0)
    im2 = axes[0, 1].imshow(weight_stds, cmap='viridis')
    axes[0, 1].set_title('Weight Std Deviations')
    plt.colorbar(im2, ax=axes[0, 1])

    # Effective sample size for weights
    try:
        ess_weights = az.ess(inf_obj, var_names=['weights'])['weights'].values
        ess_2d = ess_weights.reshape(n_time_basis, n_freq_basis)
        im3 = axes[0, 2].imshow(ess_2d, cmap='viridis')
        axes[0, 2].set_title('Effective Sample Size')
        plt.colorbar(im3, ax=axes[0, 2])
    except:
        axes[0, 2].text(0.5, 0.5, 'ESS calculation failed', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('ESS (failed)')

    # Weight trace for a few selected parameters
    selected_weights = weights_2d_samples[:, ::2, ::2]  # Subsample for display
    for i in range(min(3, selected_weights.shape[1])):
        for j in range(min(3, selected_weights.shape[2])):
            axes[1, 0].plot(selected_weights[:, i, j], alpha=0.7, label=f'w[{i},{j}]')
    axes[1, 0].set_title('Selected Weight Traces')
    axes[1, 0].set_xlabel('MCMC Iteration')
    axes[1, 0].legend()

    # Weight autocorrelation
    weight_flat = weights_samples.T  # Shape: (n_params, n_samples)
    selected_params = weight_flat[::max(1, len(weight_flat) // 10)]  # Select ~10 parameters

    for i, param_trace in enumerate(selected_params[:5]):  # Max 5 traces
        autocorr = jnp.correlate(param_trace - jnp.mean(param_trace),
                                 param_trace - jnp.mean(param_trace), mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]
        lags = jnp.arange(len(autocorr))
        axes[1, 1].plot(lags[:min(100, len(lags))], autocorr[:min(100, len(autocorr))],
                        alpha=0.7, label=f'Param {i}')

    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Weight Autocorrelations')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].legend()

    # Smoothing parameter evolution
    phi_samples = samples['phi']
    delta_samples = samples['delta']

    axes[1, 2].plot(phi_samples, label='φ', alpha=0.8)
    axes[1, 2].plot(delta_samples, label='δ', alpha=0.8)
    axes[1, 2].set_title('Smoothing Parameters')
    axes[1, 2].set_xlabel('MCMC Iteration')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.suptitle('Weight and Parameter Diagnostics', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_weights.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Weight diagnostics saved to {save_prefix}_weights.png")
    print(f"  Weight range: [{jnp.min(weight_means):.4f}, {jnp.max(weight_means):.4f}]")
    print(f"  Weight std range: [{jnp.min(weight_stds):.4f}, {jnp.max(weight_stds):.4f}]")
    print(f"  φ mean: {jnp.mean(phi_samples):.3f} ± {jnp.std(phi_samples):.3f}")
    print(f"  δ mean: {jnp.mean(delta_samples):.3f} ± {jnp.std(delta_samples):.3f}")

    return weight_means, weight_stds


def plot_fit_comparison(spectrogram: Spectrogram, model: TimeVaryingLogPSplines, mcmc: MCMC,
                        save_path="fit_comparison.png"):
    """Compare optimization vs MCMC fits."""

    # Initial fit (optimization)
    initial_log_surface = model.time_basis @ model.weights @ model.freq_basis.T
    initial_surface = jnp.exp(initial_log_surface)

    # MCMC fit
    samples = mcmc.get_samples()
    mean_weights = jnp.mean(samples['weights'], axis=0).reshape(model.n_time_basis, model.n_freq_basis)
    mcmc_log_surface = model.time_basis @ mean_weights @ model.freq_basis.T
    mcmc_surface = jnp.exp(mcmc_log_surface)

    # Compute MSEs
    log_data = jnp.log(spectrogram.power)
    initial_mse = jnp.mean((log_data - initial_log_surface) ** 2)
    mcmc_mse = jnp.mean((log_data - mcmc_log_surface) ** 2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Top row: surfaces
    im1 = axes[0, 0].pcolormesh(spectrogram.freqs, spectrogram.times, spectrogram.power, shading='auto')
    axes[0, 0].set_title('Original Data')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(spectrogram.freqs, spectrogram.times, initial_surface, shading='auto')
    axes[0, 1].set_title(f'Optimization Fit\n(Log MSE: {initial_mse:.4f})')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].pcolormesh(spectrogram.freqs, spectrogram.times, mcmc_surface, shading='auto')
    axes[0, 2].set_title(f'MCMC Fit\n(Log MSE: {mcmc_mse:.4f})')
    plt.colorbar(im3, ax=axes[0, 2])

    # Bottom row: residuals and differences
    initial_residuals = spectrogram.power - initial_surface
    mcmc_residuals = spectrogram.power - mcmc_surface
    fit_difference = mcmc_surface - initial_surface

    im4 = axes[1, 0].pcolormesh(spectrogram.freqs, spectrogram.times, initial_residuals, shading='auto', cmap='RdBu_r')
    axes[1, 0].set_title('Optimization Residuals')
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].pcolormesh(spectrogram.freqs, spectrogram.times, mcmc_residuals, shading='auto', cmap='RdBu_r')
    axes[1, 1].set_title('MCMC Residuals')
    plt.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].pcolormesh(spectrogram.freqs, spectrogram.times, fit_difference, shading='auto', cmap='RdBu_r')
    axes[1, 2].set_title('MCMC - Optimization')
    plt.colorbar(im6, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Time')

    plt.suptitle(f'Optimization vs MCMC Comparison\nMSE Ratio: {mcmc_mse / initial_mse:.3f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Fit comparison saved to {save_path}")
    print(f"  Optimization log MSE: {initial_mse:.6f}")
    print(f"  MCMC log MSE: {mcmc_mse:.6f}")
    print(f"  MSE ratio (MCMC/Opt): {mcmc_mse / initial_mse:.3f}")

    return initial_mse, mcmc_mse


def run_2d_mcmc_kronecker(
        spectrogram: Spectrogram,
        n_time_knots: int = 5,
        n_freq_knots: int = 7,
        time_degree: int = 2,
        freq_degree: int = 2,
        num_warmup=500,
        num_samples=300,
        rng_key=0,
        save_prefix="analysis"
) -> Tuple[MCMC, TimeVaryingLogPSplines]:
    """Run MCMC with Kronecker penalty and comprehensive diagnostics."""

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

    print(f"Model: {model.n_time_basis} × {model.n_freq_basis} = {model.n_time_basis * model.n_freq_basis} parameters")

    # 1. ANALYZE PENALTY MATRIX
    eigenvals, eff_rank = plot_penalty_diagnostics(model, f"{save_prefix}_penalty.png")

    # 2. INITIALIZE AND PLOT INITIAL FIT
    print("Optimizing initial weights...")
    model.weights = init_2d_weights(jnp.log(spectrogram.power), model)
    plot_initial_fit(spectrogram, model, f"{save_prefix}_initial.png")

    # 3. RUN MCMC
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
        1.0, 1.0, 1e-4, 1e-4
    )

    runtime = time.time() - t0
    print(f"MCMC completed in {runtime:.2f}s")

    # 4. CHECK DIVERGENCES
    divergences = mcmc.get_extra_fields()['diverging']
    n_divergent = jnp.sum(divergences)
    print(f"Divergences: {n_divergent}/{len(divergences)} ({100 * n_divergent / len(divergences):.1f}%)")

    # 5. PLOT ALL DIAGNOSTICS
    plot_mcmc_diagnostics(mcmc, model, save_prefix)
    plot_fit_comparison(spectrogram, model, mcmc, f"{save_prefix}_comparison.png")

    return mcmc, model


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
    """Main example with comprehensive saved diagnostics."""
    print("Generating test data...")
    spectrogram = generate_test_data()

    print("Running 2D P-spline MCMC with Kronecker penalty...")
    mcmc, model = run_2d_mcmc_kronecker(
        spectrogram,
        n_time_knots=4,
        n_freq_knots=6,
        num_warmup=400,
        num_samples=200,
        save_prefix="kronecker_analysis"
    )

    print("\n=== ALL PLOTS SAVED ===")
    print("Files created:")
    print("  - kronecker_analysis_penalty.png (penalty matrix diagnostics)")
    print("  - kronecker_analysis_initial.png (initial optimization fit)")
    print("  - kronecker_analysis_trace.png (MCMC trace plots)")
    print("  - kronecker_analysis_energy.png (energy diagnostics)")
    print("  - kronecker_analysis_weights.png (weight diagnostics)")
    print("  - kronecker_analysis_comparison.png (optimization vs MCMC)")

    return mcmc, model


if __name__ == "__main__":
    mcmc, model = main()