from torch.cuda import is_available

DEVICE = 'cuda' if is_available() else 'cpu'
