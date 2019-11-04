import yaml

class Struct:
    def __init__(self, indent=0, **entries):
        self.__dict__.update(entries)
        self._indent = indent

    def __repr__(self):
        out = []
        for k in self.__dict__:
            if not k.startswith('_'):
                out.append('{}: {}'.format(k, getattr(self, k)))
        if self._indent:
            out = [' ' * self._indent + x for x in out]
        return '\n' + '\n'.join(out)

    def _get(self, k, default=None):
        return self.__dict__.get(k, default)

    def __setitem__(self, k, v):
        self.__dict__[k] = v


with open('config.yaml') as f:
    cf = yaml.load(f, Loader=yaml.FullLoader)

audio_cf = Struct(**cf['audio'], indent=2)
sensitivity_cf = Struct(**cf['sensitivity'], indent=2)
flask_cf = Struct(**cf['flask'], indent=2)

if isinstance(flask_cf.path, str):
    flask_cf.path = [flask_cf.path]
flask_cf.model_cfs = {}
for x in flask_cf.path:
    with open(f'checkpoints/{x}/config.yaml') as f:
        flask_cf.model_cfs[x] = yaml.load(f, Loader=yaml.FullLoader)['generator']
        flask_cf.model_cfs[x]['path'] = 'checkpoints/' + x + '/model.pt'
        flask_cf.model_cfs[x] = Struct(**flask_cf.model_cfs[x], indent=2)

cf = Struct(model=flask_cf.model_cfs[flask_cf.path[0]],
            audio=audio_cf,
            sensitivity=sensitivity_cf,
            flask=flask_cf)

print(cf)
