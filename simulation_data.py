import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')


class TimeVaryingPSDSimulator:
    """
    Implements time-varying Power Spectral Density simulation based on
    locally stationary processes from Tang et al. (2024).

    Supports the three data generating processes (DGPs) from the paper:
    - LS1: Time-varying MA process with sinusoidal coefficients
    - LS2: Time-varying MA process with cosine coefficients
    - LS3: Time-varying AR process
    """

    def __init__(self, T=1500, innovation_type='normal'):
        """
        Initialize simulator.

        Parameters:
        -----------
        T : int
            Length of time series (default: 1500 as in paper)
        innovation_type : str
            Type of innovations: 'normal', 'student_t', 'pareto'
        """
        self.T = T
        self.innovation_type = innovation_type
        self.time_grid = np.linspace(0, 1, T)

    def generate_innovations(self):
        """Generate innovations according to specified distribution."""
        if self.innovation_type == 'normal':
            # i.i.d. standard normal
            return np.random.standard_normal(self.T)
        elif self.innovation_type == 'student_t':
            # Standardized Student's t with 3 degrees of freedom
            innovations = np.random.standard_t(3, self.T)
            return innovations / np.sqrt(3)  # Standardize to unit variance
        elif self.innovation_type == 'pareto':
            # Standardized Pareto distribution (scale=1, shape=4)
            innovations = np.random.pareto(4, self.T) + 1
            # Standardize to unit variance (Pareto(4) has variance 4/9)
            return (innovations - 4 / 3) / np.sqrt(4 / 9)
        else:
            raise ValueError("innovation_type must be 'normal', 'student_t', or 'pareto'")

    def simulate_LS1(self):
        """
        Simulate LS1: Time-varying MA process
        X_{t,T} = w_t + 1.122(1 - 1.718*sin(π/2 * t/T)) * w_{t-1} - 0.81 * w_{t-2}
        """
        w = self.generate_innovations()
        X = np.zeros(self.T)

        for t in range(self.T):
            X[t] = w[t]
            if t >= 1:
                ma1_coef = 1.122 * (1 - 1.718 * np.sin(np.pi / 2 * t / self.T))
                X[t] += ma1_coef * w[t - 1]
            if t >= 2:
                X[t] -= 0.81 * w[t - 2]

        return X, w

    def simulate_LS2(self):
        """
        Simulate LS2: Time-varying MA process
        X_{t,T} = w_t + 1.1*cos(1.5 - cos(4π * t/T)) * w_{t-1}
        """
        w = self.generate_innovations()
        X = np.zeros(self.T)

        for t in range(self.T):
            X[t] = w[t]
            if t >= 1:
                ma1_coef = 1.1 * np.cos(1.5 - np.cos(4 * np.pi * t / self.T))
                X[t] += ma1_coef * w[t - 1]

        return X, w

    def simulate_LS3(self):
        """
        Simulate LS3: Time-varying AR process
        X_{t,T} = (1.2 * t/T - 0.6) * X_{t-1,T} + w_t
        """
        w = self.generate_innovations()
        X = np.zeros(self.T)

        for t in range(self.T):
            X[t] = w[t]
            if t >= 1:
                ar_coef = 1.2 * t / self.T - 0.6
                X[t] += ar_coef * X[t - 1]

        return X, w

    def simulate_PS1(self):
        """
        Simulate PS1: Piecewise stationary AR process
        X_{t,T} = a_{t,T} * X_{t-1,T} + w_t
        where a_{t,T} = -0.5 if t ≤ T/2, +0.5 if t > T/2
        """
        w = self.generate_innovations()
        X = np.zeros(self.T)

        for t in range(self.T):
            X[t] = w[t]
            if t >= 1:
                ar_coef = -0.5 if t <= self.T // 2 else 0.5
                X[t] += ar_coef * X[t - 1]

        return X, w

    def compute_true_psd_LS1(self, freq_grid=None):
        """
        Compute theoretical time-varying PSD for LS1 process.

        For MA process: f(u,λ) = (1/2π) * |∑_{j} a(u,j) * exp(-iπλj)|²
        """
        if freq_grid is None:
            freq_grid = np.linspace(0, 1, 100)

        time_grid = np.linspace(0, 1, 100)
        psd = np.zeros((len(time_grid), len(freq_grid)))

        for i, u in enumerate(time_grid):
            for j, lam in enumerate(freq_grid):
                # MA coefficients for LS1
                a0 = 1.0
                a1 = 1.122 * (1 - 1.718 * np.sin(np.pi / 2 * u))
                a2 = -0.81

                # Compute |∑_{j} a_j * exp(-iπλj)|²
                freq_response = (a0 +
                                 a1 * np.exp(-1j * np.pi * lam * 1) +
                                 a2 * np.exp(-1j * np.pi * lam * 2))
                psd[i, j] = (1 / (2 * np.pi)) * np.abs(freq_response) ** 2

        return psd, time_grid, freq_grid

    def compute_true_psd_LS2(self, freq_grid=None):
        """Compute theoretical time-varying PSD for LS2 process."""
        if freq_grid is None:
            freq_grid = np.linspace(0, 1, 100)

        time_grid = np.linspace(0, 1, 100)
        psd = np.zeros((len(time_grid), len(freq_grid)))

        for i, u in enumerate(time_grid):
            for j, lam in enumerate(freq_grid):
                # MA coefficients for LS2
                a0 = 1.0
                a1 = 1.1 * np.cos(1.5 - np.cos(4 * np.pi * u))

                # Compute |∑_{j} a_j * exp(-iπλj)|²
                freq_response = a0 + a1 * np.exp(-1j * np.pi * lam * 1)
                psd[i, j] = (1 / (2 * np.pi)) * np.abs(freq_response) ** 2

        return psd, time_grid, freq_grid

    def compute_true_psd_LS3(self, freq_grid=None):
        """
        Compute theoretical time-varying PSD for LS3 process.

        For AR process: f(u,λ) = σ² / |1 - ∑_{j} φ_j * exp(-iπλj)|²
        """
        if freq_grid is None:
            freq_grid = np.linspace(0, 1, 100)

        time_grid = np.linspace(0, 1, 100)
        psd = np.zeros((len(time_grid), len(freq_grid)))

        for i, u in enumerate(time_grid):
            for j, lam in enumerate(freq_grid):
                # AR coefficient for LS3
                phi1 = 1.2 * u - 0.6

                # Compute σ² / |1 - φ₁ * exp(-iπλ)|²
                denominator = 1 - phi1 * np.exp(-1j * np.pi * lam * 1)
                psd[i, j] = (1 / (2 * np.pi)) / np.abs(denominator) ** 2

        return psd, time_grid, freq_grid

    def compute_empirical_psd(self, X, window_size=256, overlap=0.5):
        """
        Compute empirical time-varying PSD using short-time Fourier transform.

        Parameters:
        -----------
        X : array
            Time series data
        window_size : int
            Window size for STFT
        overlap : float
            Overlap fraction between windows
        """
        nperseg = window_size
        noverlap = int(overlap * nperseg)

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(X, fs=1.0, nperseg=nperseg,
                                       noverlap=noverlap,
                                       scaling='density')

        # Rescale time and frequency to [0,1]
        t_rescaled = t / t[-1] if len(t) > 1 else np.array([0])
        f_rescaled = f / f[-1] if len(f) > 1 and f[-1] > 0 else f

        return Sxx.T, t_rescaled, f_rescaled

    def plot_comparison(self, dgp_name, save_fig=False):
        """
        Generate comparison plots of true vs empirical PSD.

        Parameters:
        -----------
        dgp_name : str
            'LS1', 'LS2', 'LS3', or 'PS1'
        save_fig : bool
            Whether to save the figure
        """
        # Simulate data
        if dgp_name == 'LS1':
            X, w = self.simulate_LS1()
            true_psd, t_true, f_true = self.compute_true_psd_LS1()
        elif dgp_name == 'LS2':
            X, w = self.simulate_LS2()
            true_psd, t_true, f_true = self.compute_true_psd_LS2()
        elif dgp_name == 'LS3':
            X, w = self.simulate_LS3()
            true_psd, t_true, f_true = self.compute_true_psd_LS3()
        elif dgp_name == 'PS1':
            X, w = self.simulate_PS1()
            # For PS1, we'll compute empirical PSD only
            true_psd, t_true, f_true = None, None, None
        else:
            raise ValueError("dgp_name must be 'LS1', 'LS2', 'LS3', or 'PS1'")

        # Compute empirical PSD
        emp_psd, t_emp, f_emp = self.compute_empirical_psd(X)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{dgp_name} Process: Time-Varying PSD Analysis', fontsize=16)

        # Plot time series
        axes[0, 0].plot(X)
        axes[0, 0].set_title('Simulated Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot innovations
        axes[0, 1].plot(w)
        axes[0, 1].set_title(f'Innovations ({self.innovation_type})')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot true PSD (if available)
        if true_psd is not None:
            im1 = axes[1, 0].imshow(true_psd, aspect='auto', origin='lower',
                                    extent=[f_true[0], f_true[-1], t_true[0], t_true[-1]],
                                    cmap='viridis')
            axes[1, 0].set_title('True Time-Varying PSD')
            axes[1, 0].set_xlabel('Rescaled Frequency')
            axes[1, 0].set_ylabel('Rescaled Time')
            plt.colorbar(im1, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, 'True PSD\nNot Available\nfor PS1',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('True Time-Varying PSD')

        # Plot empirical PSD
        im2 = axes[1, 1].imshow(emp_psd, aspect='auto', origin='lower',
                                extent=[f_emp[0], f_emp[-1], t_emp[0], t_emp[-1]],
                                cmap='viridis')
        axes[1, 1].set_title('Empirical Time-Varying PSD')
        axes[1, 1].set_xlabel('Rescaled Frequency')
        axes[1, 1].set_ylabel('Rescaled Time')
        plt.colorbar(im2, ax=axes[1, 1])

        plt.tight_layout()

        if save_fig:
            plt.savefig(f'{dgp_name}_{self.innovation_type}_analysis.png',
                        dpi=300, bbox_inches='tight')

        plt.show()

        return X, w, true_psd, emp_psd


def run_simulation_study():
    """
    Run the complete simulation study for all DGPs and innovation types.
    """
    dgps = ['LS1', 'LS2', 'LS3', 'PS1']
    innovation_types = ['normal', 'student_t', 'pareto']

    results = {}

    print("Running Time-Varying PSD Simulation Study")
    print("=" * 50)

    for dgp in dgps:
        print(f"\nProcessing {dgp}...")
        results[dgp] = {}

        for innov_type in innovation_types:
            print(f"  - {innov_type} innovations")

            # Set random seed for reproducibility
            np.random.seed(42)

            # Create simulator
            sim = TimeVaryingPSDSimulator(T=1500, innovation_type=innov_type)

            # Generate data and plots
            X, w, true_psd, emp_psd = sim.plot_comparison(dgp)

            # Store results
            results[dgp][innov_type] = {
                'time_series': X,
                'innovations': w,
                'true_psd': true_psd,
                'empirical_psd': emp_psd,
                'simulator': sim
            }

    return results


def compute_average_square_error(true_psd, estimated_psd, time_grid, freq_grid):
    """
    Compute Average Square Error (ASE) as defined in equation (4.1) of the paper.

    ASE = (1/(T*(K+1))) * ∑∑[ln(f̂(t/T, j/K)) - ln(f₀(t/T, j/K))]²
    """
    if true_psd is None or estimated_psd is None:
        return np.nan

    # Interpolate to common grid if needed
    if true_psd.shape != estimated_psd.shape:
        from scipy.interpolate import RegularGridInterpolator

        # Create interpolation function for estimated PSD
        # RegularGridInterpolator expects (time, freq) order
        est_time_grid = np.linspace(0, 1, estimated_psd.shape[0])
        est_freq_grid = np.linspace(0, 1, estimated_psd.shape[1])

        est_interp = RegularGridInterpolator((est_time_grid, est_freq_grid),
                                             estimated_psd,
                                             method='linear',
                                             bounds_error=False,
                                             fill_value=None)

        # Create meshgrid for evaluation points
        T_mesh, F_mesh = np.meshgrid(time_grid, freq_grid, indexing='ij')
        eval_points = np.column_stack([T_mesh.ravel(), F_mesh.ravel()])

        # Evaluate on true PSD grid
        estimated_psd_interp = est_interp(eval_points).reshape(true_psd.shape)
    else:
        estimated_psd_interp = estimated_psd

    # Avoid log of zero by adding small epsilon
    eps = 1e-10
    log_true = np.log(true_psd + eps)
    log_est = np.log(estimated_psd_interp + eps)

    # Compute ASE
    ase = np.mean((log_est - log_true) ** 2)

    return ase


def demonstrate_gravitational_wave_application():
    """
    Demonstrate application to gravitational wave data analysis.
    This shows how the methodology could be applied to LIGO-type data.
    """
    print("\nGravitational Wave Application Demonstration")
    print("=" * 50)

    # Simulate LIGO-like noise characteristics
    # Higher sampling rate, longer duration
    T_gw = 4096  # 1 second at 4096 Hz (typical LIGO sampling)

    # Create simulator with different characteristics
    sim_gw = TimeVaryingPSDSimulator(T=T_gw, innovation_type='normal')

    # Generate LS2-type process (good for demonstrating non-stationarity)
    X_gw, w_gw = sim_gw.simulate_LS2()

    # Add some realistic scaling
    X_gw = X_gw * 1e-18  # Scale to strain-like amplitudes

    # Compute time-varying PSD with higher resolution
    emp_psd_gw, t_gw, f_gw = sim_gw.compute_empirical_psd(X_gw, window_size=128, overlap=0.75)

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Gravitational Wave Detector Noise Simulation', fontsize=14)

    # Time series
    time_axis = np.linspace(0, 1, T_gw)
    axes[0].plot(time_axis, X_gw)
    axes[0].set_title('Simulated Detector Strain')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Strain')
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Time-varying PSD
    im = axes[1].imshow(emp_psd_gw.T, aspect='auto', origin='lower',
                        extent=[t_gw[0], t_gw[-1], f_gw[0] * 2048, f_gw[-1] * 2048],
                        cmap='plasma')
    axes[1].set_title('Time-Varying Power Spectral Density')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_yscale('log')
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('PSD (strain²/Hz)')

    plt.tight_layout()
    plt.show()

    return X_gw, emp_psd_gw, t_gw, f_gw


if __name__ == "__main__":
    # Run basic demonstration
    print("Time-Varying PSD Simulator")
    print("Based on Tang et al. (2024): Bayesian nonparametric spectral analysis")
    print("of locally stationary processes")
    print()

    # Quick example with LS2
    np.random.seed(42)
    sim = TimeVaryingPSDSimulator(T=1500, innovation_type='normal')

    print("Generating LS2 example...")
    X, w, true_psd, emp_psd = sim.plot_comparison('LS2')

    # Compute ASE if true PSD is available
    if true_psd is not None:
        ase = compute_average_square_error(true_psd, emp_psd,
                                           np.linspace(0, 1, true_psd.shape[0]),
                                           np.linspace(0, 1, true_psd.shape[1]))
        print(f"Average Square Error: {ase:.4f}")

    # Uncomment to run full study:
    # results = run_simulation_study()

    # Uncomment for gravitational wave demonstration:
    # demonstrate_gravitational_wave_application()