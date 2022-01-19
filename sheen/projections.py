import pickle

import numpy
import sklearn.decomposition


class Projection:
    """
    Projection.
    """
    def update(self, x):
        return x

    def transform(self, x):
        return self._model.transform(x)


class IdentityProjection(Projection):
    def __init__(self, full_dimension):
        """
        Identity mapping.

        :param full_dimension: full dimension.
        """
        self.n_components = full_dimension
        self._model = lambda x: x


class StaticICA(Projection):
    def __init__(self, path_):
        """
        Pre-calibrated ICA transformation.

        :param path_: path where sklearn model is saved.
        """
        self.path_ = path_
        with open(path_, 'rb') as f:
            self._model = pickle.load(f)
        self.n_components = self._model.n_components


class IncrementalPCA(Projection):
    def __init__(self, n_components=20, projection_window=2000):
        """
        Incremental PCA updated which each incoming sample.

        :param n_components: number of components.
        :param projection_window: incremental PCA is allowed to change over time with this setting.
        """
        self.projection_window = projection_window
        self.n_components = n_components
        self._model = sklearn.decomposition.IncrementalPCA(n_components=n_components)

    def update(self, x):
        self._model.partial_fit(x)
        self._model.n_samples_seen_ = min(
            self._model.n_samples_seen_,
            self.projection_window,
        )

