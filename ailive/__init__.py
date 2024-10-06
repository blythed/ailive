from torch.backends.mps import is_available

DEVICE = 'mps' if is_available() else 'cpu'

print('DEVICE:', DEVICE)
