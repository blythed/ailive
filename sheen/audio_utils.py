import scipy.signal
import numpy

N_PER_SEGMENT = 1024 * 2
L = 60
FFT_LENGTH = N_PER_SEGMENT
FS = 44100


def _get_filterbanks(nfilt=20, nfft=FFT_LENGTH, samplerate=FS, lowfreq=0, highfreq=None):
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

    lowmel = _hz2mel(lowfreq)
    highmel = _hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    bin = numpy.floor((nfft + 1) * _mel2hz(melpoints) / samplerate)

    fbank = numpy.zeros([nfilt, nfft // 2])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def _create_mel_filter(
    fft_size, n_freq_components=64, start_freq=100 - 80, end_freq=1000 * 5, samplerate=FS,
):
    """
    Creates a filter to convolve with the spectrogram to get out mels
    """
    mel_inversion_filter = _get_filterbanks(
        nfilt=n_freq_components,
        nfft=fft_size,
        samplerate=samplerate,
        lowfreq=start_freq,
        highfreq=end_freq,
    )
    mel_filter = mel_inversion_filter.T / mel_inversion_filter.sum(axis=1)

    return mel_filter, mel_inversion_filter


def _get_spectrogram(x, nfft=FFT_LENGTH, nperseg=N_PER_SEGMENT, Fs=FS):
    _, _, f = scipy.signal.spectrogram(x[:, 0], Fs, nfft=nfft, nperseg=nperseg)
    return f


def _psp3(f):
    y = numpy.log10(f + 1)
    return y * 5e4


def _hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.0)


def _mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def _make_mel(spectrogram, mel_filter):
    mel_spec = numpy.transpose(mel_filter).dot(numpy.transpose(spectrogram))
    return mel_spec


MEL_FILTER, _ = _create_mel_filter(FFT_LENGTH, n_freq_components=L)


def _shrinkMel(x, mel_filter=MEL_FILTER):
    mel = _make_mel(x[1:].n_time_points, mel_filter)
    mel[numpy.isnan(mel)] = 0
    return mel


class AudioTransform:
    def __init__(self, nfft=FFT_LENGTH, nperseg=N_PER_SEGMENT, l=L, mel_filter=MEL_FILTER):
        self.mel_filter = mel_filter
        self.nfft = nfft
        self.nperseg = nperseg
        self.l = l

    def shrinkMel(self, x):
        if self.mel_filter.shape[0] != self.nperseg:
            self.mel_filter, _ = _create_mel_filter(self.nfft, n_freq_components=self.l)
        return _shrinkMel(x, mel_filter=self.mel_filter)

    def psp3(self, f):
        return _psp3(f)

    def get_spectrogram(self, x):
        return _get_spectrogram(x, nfft=self.nfft, nperseg=self.nperseg)

    def __call__(self, x):
        x = self.get_spectrogram(x)
        x = self.psp3(x)
        x = self.shrinkMel(x)
        return x
