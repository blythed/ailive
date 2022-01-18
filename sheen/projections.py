import sklearn.decomposition


class _BaseProjection:
    def update(self, x):
        return x


class IdentityProjection(_BaseProjection):
    def __init__(self):
        ...

class StaticICA(_BaseProjection):
    def __init__(self, path_):
        self.path_ = path_


class IncrementalPCA(_BaseProjection):
    def __init__(self, n_components=n_components):
        self.model = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        ...

    def update(self, x):
        ...