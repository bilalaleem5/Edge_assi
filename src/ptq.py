"""Post-Training Static Quantization (PTQ) of the FP32 ResNet-18 -> INT8.
Steps: load FP32 -> fuse Conv+BN+ReLU -> set backend -> insert observers -> calibrate -> convert -> save."""
import argparse, os, copy, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--fp32', default=os.path.join(MODEL_DIR, 'resnet18_fp32.pth'))
ap.add_argument('--out', default=os.path.join(MODEL_DIR, 'resnet18_ptq_int8.pt'))
ap.add_argument('--calib-size', type=int, default=2000, help='number of training images used for calibration')
ap.add_argument('--backend', default='fbgemm')
a = ap.parse_args()

set_seed(); torch.set_num_threads(os.cpu_count())
torch.backends.quantized.engine = a.backend                      # 1) configure quantization backend
xt, yt = load_split('test'); x, y = load_split('train')

model = ResNet18(); model.load_state_dict(torch.load(a.fp32)); model.eval()
print(f'FP32 accuracy: {evaluate(model, xt, yt):.2f}%')

model.fuse_model()                                                # 2) fuse Conv+BN(+ReLU)
model.qconfig = torch.ao.quantization.get_default_qconfig(a.backend)   # 3) static qconfig (per-channel weights, histogram/minmax act.)
torch.ao.quantization.prepare(model, inplace=True)                # 4) insert observers

g = torch.Generator().manual_seed(SEED)                           # 5) calibration on a random subset of TRAIN data
idx = torch.randperm(x.size(0), generator=g)[:a.calib_size]
with torch.no_grad():
    for xb, _ in batches(x[idx], y[idx], 100):
        model(xb)
print(f'Calibrated on {a.calib_size} training images')

torch.ao.quantization.convert(model, inplace=True)                # 6) FP32 -> INT8
acc = evaluate(model, xt, yt)
print(f'PTQ INT8 accuracy: {acc:.2f}%')

scripted = torch.jit.script(model)                                # 7) deployable artefact (TorchScript, no Python class needed)
torch.jit.save(scripted, a.out)
torch.save(model.state_dict(), a.out.replace('.pt', '_state_dict.pth'))
print(f'Saved {a.out} ({file_mb(a.out):.2f} MB)')
