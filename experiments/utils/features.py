import numpy as np
from scipy.signal import stft, periodogram, welch
from scipy.stats import entropy
import pywt


def calc_damv_all(data):
    return np.mean(np.abs(np.diff(data, axis=1)), axis=1)

def calc_mad(data):
    median = np.median(data, axis=1)
    mad = np.median(np.abs(data - median[:, None]), axis=1)
    return mad

def RootMeanSqure(data):
    return np.sqrt(np.mean(data ** 2, axis=1))

def MeanAbsoluteValue(data):
    return np.mean(np.abs(data), axis=1)

def Zerocrossing(data):
    return np.sum(np.abs(np.diff(np.sign(data), axis=1)), axis=1)/2

def SlopeSignChanges(data):
    return np.sum(np.abs(np.diff(np.sign(np.sign(np.diff(data, axis=1))), axis=1)), axis=1)/2

def WaveformLength(data):
    return np.sum(np.abs(np.diff(data, axis=1)), axis=1)

def SimpleSquareIntegral(data):
    return np.sum(np.square(data), axis=1)

def Variance(data):
    return np.sum(np.square(data), axis=1)/(data.shape[1]-1)

def LogDetector(data):
    return np.exp(np.mean(np.log((np.abs(data))), axis=1))

def AverageAmplitudeChange(data):
    return np.sum(np.abs(np.diff(data, axis=1)), axis=1) / (data.shape[1])

def DifferenceAbsoluteDeviation(data):
    return np.sqrt(np.sum(np.abs(np.diff(data, axis=1)), axis=1)/(data.shape[1]))

def StatMaximum(data):
    return np.max(data, axis=1)

def StatMin(data):
    return np.min(data, axis=1)

def StatStd(data):
    return np.std(data, axis=1, ddof=1)

def StatMean(data):
    return np.mean(data, axis=1)

def RootSquaredZeroOrderMoment(data):
    return np.sqrt(np.sum(np.square(data), axis=1))

def RootSquaredSecondOrderMoment(data):
    return np.sqrt(np.mean(np.diff(data, axis=1) ** 2, axis=1))

def RootSquaredFourthOrderMoment(data):
    return np.sqrt(np.mean(np.diff(np.diff(data, axis=1), axis=1) ** 2, axis=1))

def nRootSquaredZeroOrderMoment(data):
    return (RootSquaredZeroOrderMoment(data) ** 0.1) / 0.1

def nRootSquaredSecondOrderMoment(data):
    return (RootSquaredSecondOrderMoment(data) ** 0.1) / 0.1

def nRootSquaredFourthOrderMoment(data):
    return (RootSquaredFourthOrderMoment(data) ** 0.1) / 0.1

def Sparseness(data):
    return nRootSquaredZeroOrderMoment(data)/(np.sqrt(nRootSquaredZeroOrderMoment(data) - nRootSquaredSecondOrderMoment(data)) *
                                              np.sqrt(nRootSquaredZeroOrderMoment(data) - nRootSquaredFourthOrderMoment(data)))

def WaveformLengthRatio(data):
    return np.sum(np.abs(np.diff(data, axis=1)), axis=1)/np.sum(np.abs(np.diff(np.diff(data, axis=1), axis=1)), axis=1)

def TDPSD_f1(data):
    return np.log(nRootSquaredZeroOrderMoment(data))

def TDPSD_f2(data):
    return np.log(nRootSquaredZeroOrderMoment(data)-nRootSquaredSecondOrderMoment(data))

def TDPSD_f3(data):
    return np.log(nRootSquaredZeroOrderMoment(data)-nRootSquaredFourthOrderMoment(data))

def invTDD_f1(data):
    return (np.sum(np.square(data), axis=1) ** 0.1) / 0.1

def invTDD_f2(data):
    return nRootSquaredFourthOrderMoment(data)/(nRootSquaredSecondOrderMoment(data)*invTDD_f1(data))

def ZeroOrderMoment(data):
    return np.log((np.sqrt(np.sum(np.square(data), axis=1)) ** 0.1) / 0.1)

def SecondOrderMoment(data):
    return (np.sqrt(np.mean(np.diff(data, axis=1) ** 2, axis=1)) ** 0.1) / 0.1

def FourthOrderMoment(data):
    return (np.sqrt(np.mean(np.diff(np.diff(data, axis=1), axis=1) ** 2, axis=1)) ** 0.1) / 0.1

def PeakStress(data):
    return FourthOrderMoment(data)/(SecondOrderMoment(data)*ZeroOrderMoment(data))

def MeanAbsoluteSecondOrderDiffi(data):
    return (np.mean(np.abs(np.diff(np.diff(data, axis=1), axis=1)), axis=1) ** 0.1) / 0.1

def MeanLog(data):
    return (np.abs(np.log(np.mean(np.square(data), axis=1))) ** 0.1) / 0.1

def DoffStd(data):
    return np.std(data, axis=1, ddof=1)

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def calc_spectral_rolloff_all(data, fs=4000, rolloff_percent=0.85):
    """
    Calculate spectral roll-off for each channel.
    The roll-off frequency is where cumulative power reaches rolloff_percent of total.
    Returns: array shape (channels,)
    """
    rolloffs = []
    for signal in data:
        freqs, psd = welch(signal, fs=fs)
        cumulative = np.cumsum(psd)
        threshold = rolloff_percent * cumulative[-1]
        rolloff_freq = freqs[np.where(cumulative >= threshold)[0][0]]
        rolloffs.append(rolloff_freq)
    return np.array(rolloffs)


def _phi(x, m, r):
    N = len(x)
    # Count matches for each template vector
    C = []
    for i in range(N - m + 1):
        template = x[i:i + m]
        count = np.sum([1 for j in range(N - m + 1)
                        if np.max(np.abs(template - x[j:j + m])) <= r])
        C.append(count / (N - m + 1))
    return np.sum(np.log(C)) / (N - m + 1)


def calc_approx_entropy_all(data, m=2, r_factor=0.2):
    """
    Calculate Approximate Entropy for each channel.
    m: embedding dimension
    r_factor: tolerance = r_factor * std(signal)
    Returns: array shape (channels,)
    """
    apen = []
    for signal in data:
        r = r_factor * np.std(signal)
        phi_m = _phi(signal, m, r)
        phi_m1 = _phi(signal, m + 1, r)
        apen.append(phi_m - phi_m1)
    return np.array(apen)


def calc_sample_entropy_all(data, m=2, r_factor=0.2):
    """
    Calculate Sample Entropy for each channel.
    m: embedding dimension
    r_factor: tolerance = r_factor * std(signal)
    Returns: array shape (channels,)
    """
    sentropy = []
    for signal in data:
        r = r_factor * np.std(signal)
        N = len(signal)

        def _count_matches(m):
            count = 0
            for i in range(N - m):
                for j in range(i + 1, N - m + 1):
                    if np.max(np.abs(signal[i:i + m] - signal[j:j + m])) <= r:
                        count += 1
            return count

        B = _count_matches(m)
        A = _count_matches(m + 1)
        se = -np.log(A / B) if B != 0 and A != 0 else np.nan
        sentropy.append(se)
    return np.array(sentropy)

def petrosian_fd(signal):
    """Calculate Petrosian Fractal Dimension for one signal."""
    N = len(signal)
    diff = np.diff(signal)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)  # number of sign changes
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))

def calc_petrosian_fd_all(data):
    return np.array([petrosian_fd(channel) for channel in data])

def calc_katz_fd_al1l(data, k_max=10):
    return np.array([petrosian_fd(channel) for channel in data])

def calc_katz_fd_all(data):
    """
    Calculate Katz Fractal Dimension for each channel.
    FD = log10(N) / (log10(N) + log10(L/d))
      where N = length, L = total path length, d = max distance from first point
    Returns: array shape (channels,)
    """
    fds = []
    data=data.T
    for signal in data:
        N = len(signal)
        # cumulative path length
        diffs = np.abs(np.diff(signal))
        L = np.sum(diffs)
        d = np.max(np.abs(signal - signal[0]))
        fd = np.log10(N) / (np.log10(N) + np.log10(L / d + 1e-10))
        fds.append(fd)
    return np.array(fds)

def calc_spectral_centroid_all(data, fs=4000):
    centroids = []
    for channel in data:
        f, Pxx = periodogram(channel, fs)
        centroid = np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10)  # 防止除零
        centroids.append(centroid)
    return np.array(centroids)

def calc_spectral_bandwidth_all(data, fs=4000):
    bandwidths = []
    for channel in data:
        f, Pxx = periodogram(channel, fs)
        centroid = np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10)
        bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * Pxx) / (np.sum(Pxx) + 1e-10))
        bandwidths.append(bandwidth)
    return np.array(bandwidths)

def calc_spectral_entropy_all(data, fs=4000):
    spectral_entropy = []
    for channel in data:
        freqs, psd = welch(channel, fs=fs)
        psd_norm = psd / np.sum(psd)
        spectral_entropy.append(entropy(psd_norm))
    return np.array(spectral_entropy)  # (256,)

def calc_wavelet_energy_all(data, wavelet='db4', level=4):

    wavelet_energy = []
    for channel in data:
        coeffs = pywt.wavedec(channel, wavelet, level=level)
        detail_coeffs = coeffs[1:]
        energy = np.sum([np.sum(c**2) for c in detail_coeffs])
        wavelet_energy.append(energy)
    return np.array(wavelet_energy)  # (256,)

def calc_stft_energy_all(data, fs=4000, nperseg=128):
    stft_energy = []
    for channel in data:
        f, t, Zxx = stft(channel, fs=fs, nperseg=nperseg)
        energy = np.sum(np.abs(Zxx)**2)  # STFT
        stft_energy.append(energy)
    return np.array(stft_energy)  # (256,)

def calc_rms_all(data):
    return np.sqrt(np.mean(data**2, axis=1))

def calc_mav_all(data):
    return np.mean(np.abs(data), axis=1)

def calc_wl_all(data):
    return np.sum(np.abs(np.diff(data, axis=1)), axis=1)

def calc_zom_all(data):
    return np.sqrt(np.sum(np.square(data), axis=1))

def calc_som_all(data):
    return np.sqrt(np.mean(np.diff(data, axis=1) ** 2, axis=1))

def calc_fom_all(data):
    return np.sqrt(np.mean(np.diff(np.diff(data, axis=1), axis=1) ** 2, axis=1))

def calc_ps_all(data):
    return Sparseness(data)

def calc_logd_all(data):
    return LogDetector(data)

def calc_max_all(data):
    return StatMaximum(data)

def calc_min_all(data):
    return StatMin(data)

def calc_tdpsd_f1_all(data):
    return TDPSD_f1(data)

def calc_tdpsd_f2_all(data):
    return TDPSD_f2(data)

def calc_tdpsd_f3_all(data):
    return TDPSD_f3(data)

def calc_invtdd_f1_all(data):
    return invTDD_f1(data)

def calc_invtdd_f2_all(data):
    return invTDD_f2(data)

def calc_std_all(data):
    return StatStd(data)

def calc_ssi_all(data):
    return SimpleSquareIntegral(data)

def calc_aac_all(data):
    return AverageAmplitudeChange(data)

def calc_dasdv_all(data):
    return DifferenceAbsoluteDeviation(data)

def calc_wlr_all(data):
    return WaveformLengthRatio(data)

def calc_masod_all(data):
    return MeanAbsoluteSecondOrderDiffi(data)

def calc_ml_all(data):
    return MeanLog(data)

def calc_ds_all(data):
    return DoffStd(data)

def calc_damv_all(data):
    return np.mean(np.abs(np.diff(data, axis=1)), axis=1)

def calc_rmss_all(data):
    diff = np.diff(data, axis=1)
    return np.sqrt(np.mean(diff ** 2, axis=1))

def calc_var_all(data):
    return np.var(data, axis=1)

def calc_zc_all(data, threshold=0.01):
    signs = np.sign(data)
    crossings = signs[:, :-1] * signs[:, 1:] < 0
    valid = (np.abs(data[:, :-1]) > threshold) | (np.abs(data[:, 1:]) > threshold)
    return np.sum(crossings & valid, axis=1)

def calc_ssc_all(data, threshold=0.01):
    diff_signal = np.diff(data, axis=1)
    ssc = ((diff_signal[:, :-1] * diff_signal[:, 1:]) < 0) & \
          (np.abs(diff_signal[:, :-1]) > threshold) & (np.abs(diff_signal[:, 1:]) > threshold)
    return np.sum(ssc, axis=1)

def calc_wamp_all(data, threshold=0.1):
    return np.sum(np.abs(np.diff(data, axis=1)) > threshold, axis=1)

def calc_mnf_all(data, fs=4000):
    mnf_list = []
    for i in range(data.shape[0]):
        f, Pxx = welch(data[i], fs=fs)
        mnf = np.sum(f * Pxx) / np.sum(Pxx)
        mnf_list.append(mnf)
    return np.array(mnf_list)

def calc_mdf_all(data, fs=4000):
    mdf_list = []
    for i in range(data.shape[0]):
        f, Pxx = welch(data[i], fs=fs)
        cumulative_power = np.cumsum(Pxx)
        median_freq = f[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]]
        mdf_list.append(median_freq)
    return np.array(mdf_list)

def calc_skewness_all(data):
    fs=4000
    pkf = []
    for channel in data:
        freqs, psd = welch(channel, fs=fs, nperseg=256)
        peak_freq = freqs[np.argmax(psd)]  # 找到最大功率对应的频率
        pkf.append(peak_freq)
    return np.array(pkf)


def calc_hjorth_activity_all(data):
    return np.var(data, axis=1)

def calc_hjorth_mobility_all(data):
    diff_data = np.diff(data, axis=1)
    return np.sqrt(np.var(diff_data, axis=1) / np.var(data, axis=1))

def calc_hjorth_complexity_all(data):
    diff_data = np.diff(data, axis=1)
    diff_diff_data = np.diff(diff_data, axis=1)
    mobility = np.sqrt(np.var(diff_data, axis=1) / np.var(data, axis=1))
    mobility_diff = np.sqrt(np.var(diff_diff_data, axis=1) / np.var(diff_data, axis=1))
    return mobility_diff / mobility


def calculate_zom_multi(emg_signal):
    zero_crossings = np.diff(np.sign(emg_signal), axis=0) != 0  # 检测零交叉
    zom_values = zero_crossings.sum(axis=0) / emg_signal.shape[0]  # 每通道零交叉率
    return zom_values


def calculate_se_multi(emg_signal, bins=20):
    se_values = []
    for channel in range(emg_signal.shape[1]):
        hist, _ = np.histogram(emg_signal[:, channel], bins=bins, density=True)  # 直方图
        se = entropy(hist + 1e-10)  # 避免 log(0)
        se_values.append(se)
    return np.array(se_values)

def calculate_ustd_multi(emg_signal):
    ustd_values = np.std(emg_signal, axis=0, ddof=1)  # 按列计算无偏标准差
    return ustd_values


def calculate_rmsf(emg_data, fs=4000):
    rmsf_features = np.zeros(emg_data.shape[1])
    for channel in range(emg_data.shape[1]):
        signal = emg_data[:, channel]
        freqs, psd = welch(signal, fs=fs, nperseg=256)
        rmsf = np.sqrt(np.sum(freqs ** 2 * psd) / np.sum(psd))
        rmsf_features[channel] = rmsf
    return rmsf_features

