from ailive.train import Trainer, get_configs

import os
import sys

if not os.path.exists(f'checkpoints/{sys.argv[1]}/config.yaml'):
    os.system(f'mkdir -p checkpoints/{sys.argv[1]}')
    os.system('cp -r checkpoints/default/config.yaml'
              f' checkpoints/{sys.argv[1]}/config.yaml')

if not os.path.exists(f'data/images/{sys.argv[2]}'):
    print('copying images from downloads...')
    os.system(f'mkdir -p data/images/{sys.argv[2]}')
    os.system(f'mv /home/michaela/Downloads/*.jpg data/images/{sys.argv[2]}/')

Trainer(
    sys.argv[1],
    *get_configs(
        path='checkpoints/' + sys.argv[1] + '/config.yaml',
        dataroot='data/images/'  + sys.argv[2] + '/',
        name=sys.argv[1],
    )
).train()
