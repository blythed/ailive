from scipy import signal
import scipy
import numpy as np

NPERSEG = 1024 * 2
L = 60
NFFT = NPERSEG


def getTFFT(length, nperseg=NPERSEG):
    r = 1.0 / 8
    return (length - int(r * nperseg)) // (int(7 * r * nperseg))


def sp3(x, nfft=NFFT, nperseg=NPERSEG):
    Fs = 44100
    _, _, f = signal.spectrogram(x[:, 0], Fs, nfft=nfft, nperseg=nperseg)
    return f


def psp3(f):

    y = np.log10(f + 1)
    return y * 5e4

    y = 10 * np.log10(f + 1e-30)  # 1e-10
    return y / 200.0 * 3 + 1.7 # ????????


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def create_mel_filter(
    fft_size, n_freq_components=64, start_freq=100 - 80, end_freq=1000 * 5, samplerate=44100
):
    """
    Creates a filter to convolve with the spectrogram to get out mels

    """
    mel_inversion_filter = get_filterbanks(
        nfilt=n_freq_components,
        nfft=fft_size,
        samplerate=samplerate,
        lowfreq=start_freq,
        highfreq=end_freq,
    )
    mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

    return mel_filter, mel_inversion_filter


def make_mel(spectrogram, mel_filter, shorten_factor=1):
    mel_spec = np.transpose(mel_filter).dot(np.transpose(spectrogram))
    return mel_spec


mel_filter, _ = create_mel_filter(NFFT, n_freq_components=L)


def shrink(magnitude, S):
    xi = np.linspace(0, magnitude.shape[0] - 1, S)
    magnitude = np.interp(xi, np.arange(magnitude.shape[0]), magnitude)
    return magnitude


def shrinkMel(x, mel_filter=mel_filter):
    mel = make_mel(x[1:].T, mel_filter)
    mel[np.isnan(mel)] = 0
    return mel


class AA(object):
    def __init__(self, nfft=NFFT, nperseg=NPERSEG):
        self.mel_filter = mel_filter
        self.nfft = nfft
        self.nperseg = nperseg

    def shrinkMel(self, x):
        if self.mel_filter.shape[0] != self.nperseg:
            mel_filter, _ = create_mel_filter(self.nfft, n_freq_components=L)
        return shrinkMel(x, mel_filter=mel_filter)

    def getTFFT(self, length):
        print(self.nperseg)
        return getTFFT(length, self.nperseg)

    def sp3(self, x):
        return sp3(x, nfft=self.nfft, nperseg=self.nperseg)
