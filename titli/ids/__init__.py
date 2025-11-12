# from .kitsune import KitNET
# from .torch_kitnet import TorchKitNET
# from .pytorch_kitsune import PyTorchKitsune
from .kitnet import KitNET
from .lof import LOF
from .ocsvm import OCSVM
from .ae import Autoencoder
from .vae import VAE
from .icl import ICL

__all__ = [
    # 'KitNET',
    # 'TorchKitNET',
    # 'PyTorchKitsune',
    'KitNET',
    'LOF',
    'OCSVM',
    'Autoencoder',
    'VAE',
    'ICL'
]
