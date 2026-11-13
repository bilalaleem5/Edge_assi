"""Convert the CIFAR-10 image folders into compact uint8 numpy arrays (fast loading)."""
import os, sys, numpy as np
from PIL import Image
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
root = sys.argv[1]; out = sys.argv[2]
os.makedirs(out, exist_ok=True)
for split in ['train','test']:
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        d = os.path.join(root, split, c)
        for f in sorted(os.listdir(d), key=lambda s: int(os.path.splitext(s)[0])):
            X.append(np.array(Image.open(os.path.join(d, f)).convert('RGB'), dtype=np.uint8)); y.append(ci)
    X = np.stack(X); y = np.array(y, dtype=np.int64)
    np.savez_compressed(os.path.join(out, f'cifar10_{split}.npz'), x=X, y=y)
    print(split, X.shape, y.shape, np.bincount(y))
