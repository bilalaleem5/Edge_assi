"""Unified inference pipeline for all five PyTorch models.
   fp32 | ptq | qat | unstructured | structured
Usage:
  python src/inference.py --model all --n 10                 # predict first N test images with every model
  python src/inference.py --model ptq --image path/to/img.png # classify one image
"""
import argparse, os, torch, numpy as np
from PIL import Image
from common import *

MODELS = {
    'fp32': ('resnet18_fp32.pth', 'FP32 baseline'),
    'ptq': ('resnet18_ptq_int8.pt', 'PTQ INT8'),
    'qat': ('resnet18_qat_int8.pt', 'QAT INT8'),
    'unstructured': ('resnet18_unstructured_pruned.pth', 'Unstructured-pruned (80%)'),
    'structured': ('resnet18_structured_pruned.pth', 'Structured-pruned (channels)'),
}


def model_path(name):
    return os.path.join(MODEL_DIR, MODELS[name][0])


def load_model(name):
    torch.backends.quantized.engine = 'fbgemm'
    p = model_path(name)
    if name in ('ptq', 'qat'):
        m = torch.jit.load(p)                       # TorchScript INT8 model
    elif name == 'structured':
        ck = torch.load(p); m = ResNet18(mids=ck['mids']); m.load_state_dict(ck['state_dict'])
    else:
        m = ResNet18(); m.load_state_dict(torch.load(p))
    return m.eval()


def preprocess_image(path):
    im = Image.open(path).convert('RGB').resize((32, 32))
    return normalize(torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0))


@torch.no_grad()
def predict(model, xb):
    p = torch.softmax(model(xb), 1)
    conf, idx = p.max(1)
    return idx, conf


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='all', choices=list(MODELS) + ['all'])
    ap.add_argument('--image', default=None)
    ap.add_argument('--n', type=int, default=8, help='number of test images when no --image is given')
    a = ap.parse_args()
    torch.set_num_threads(1)
    names = list(MODELS) if a.model == 'all' else [a.model]
    if a.image:
        xb = preprocess_image(a.image); truth = None
    else:
        xt, yt = load_split('test'); xb = normalize(xt[:a.n]); truth = yt[:a.n]
    for n in names:
        idx, conf = predict(load_model(n), xb)
        print(f'--- {MODELS[n][1]} ---')
        for i in range(len(idx)):
            t = f' (true: {CLASSES[truth[i]]})' if truth is not None else ''
            print(f'  sample {i}: {CLASSES[idx[i]]:<10} conf {conf[i]:.2f}{t}')
