import h5py
import numpy as np
from pywavelet.types import TimeSeries
import glob

DIR = 'LS_sim_data'


def load_data(filepath):
    with h5py.File(filepath, 'r') as f:
        X = np.array(f['time_series'])
        w = np.array(f['innovations'])
        true_psd = np.array(f['true_psd']) if 'true_psd' in f else None
        t_true = np.array(f['t_true']) if 't_true' in f else None
        f_true = np.array(f['f_true']) if 'f_true' in f else None

    # truncate to 16384 samples if necessary
    if X.shape[0] > 16384:
        X = X[:16384]

    time = np.arange(X.shape[0])
    ts = TimeSeries(X, time)
    n = X.shape[0]
    # Nf == Nt == root(n)
    Nt = Nf = int(np.sqrt(n))
    wavelet = ts.to_wavelet(Nf, Nt)
    return X, true_psd, wavelet


def plot(X, true_psd, wavelet, fname):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].plot(X)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')

    # imshow for true PSD
    axs[1].imshow(true_psd.T, aspect='auto', origin='lower', extent=[0, true_psd.shape[0], 0, true_psd.shape[1]])
    axs[1].set_title('True PSD')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')

    wavelet.plot(ax=axs[2], cmap='viridis')
    axs[2].set_title('Wavelet Transform')

    plt.tight_layout()
    plt.savefig(fname)


if __name__ == '__main__':
    files = glob.glob(f'{DIR}/*.h5')
    for file in files:
        png_name = file.replace('.h5', '.png')
        print(f'Processing {file} -> {png_name}')
        X, true_psd, wavelet = load_data(file)
        plot(X, true_psd, wavelet, png_name)
        print(f'Saved plot to {png_name}')