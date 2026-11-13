"""Download the official CIFAR-10 with torchvision and convert it to data/cifar10_{train,test}.npz (use this on your own machine)."""
import os, numpy as np
from torchvision.datasets import CIFAR10
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'); os.makedirs(out, exist_ok=True)
for split, train in [('train', True), ('test', False)]:
    ds = CIFAR10(root=os.path.join(out, 'raw'), train=train, download=True)
    np.savez_compressed(os.path.join(out, f'cifar10_{split}.npz'), x=np.asarray(ds.data, dtype=np.uint8), y=np.asarray(ds.targets, dtype=np.int64))
    print(split, ds.data.shape)
