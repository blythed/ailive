import pickle
from scipy.io import wavfile
from ailive.utils import AA, psp3
from sklearn.decomposition import FastICA


def prep_audio(
    path,
):
    fs, x = wavfile.read(path)
    audioA = AA()
    magnitude = audioA.sp3(x[:, None])
    magnitude = psp3(magnitude)
    magnitude = audioA.shrinkMel(magnitude)
    model = FastICA(n_components=20).fit(magnitude.T)
    transformed = model.transform(magnitude.T)
    transformed = [transformed[i, :] for i in range(transformed.shape[0])]
    directory = '/'.join(path.split('/')[:-1])
    with open(directory + '/transformed.dat', 'wb') as f:
        pickle.dump(transformed, f)
    with open(directory + '/ica.model', 'wb') as f:
        pickle.dump(model, f)
