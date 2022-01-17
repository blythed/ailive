

class _BaseProjection:
    def update(self, x):
        return x


class IdentityProjection(_BaseProjection):
    def __init__(self):
        ...


class PCAProjection(_BaseProjection):
    def __init__(self, **kwargs):
        ...

    def update(self, x):
        ...