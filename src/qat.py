"""Quantization-Aware Training (QAT): FP32 checkpoint -> fuse -> fake-quant -> fine-tune -> INT8."""
import argparse, os, time, torch
import torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--fp32', default=os.path.join(MODEL_DIR, 'resnet18_fp32.pth'))
ap.add_argument('--out', default=os.path.join(MODEL_DIR, 'resnet18_qat_int8.pt'))
ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--lr', type=float, default=0.01)
ap.add_argument('--backend', default='fbgemm')
ap.add_argument('--max-batches', type=int, default=None)
a = ap.parse_args()

set_seed(); torch.set_num_threads(os.cpu_count())
torch.backends.quantized.engine = a.backend
x, y = load_split('train'); xt, yt = load_split('test')

model = ResNet18(); model.load_state_dict(torch.load(a.fp32))      # 1) initialise from the pretrained FP32 model
print(f'FP32 accuracy: {evaluate(model, xt, yt):.2f}%')
model.train(); model.fuse_model()                                  # 2) fuse (QAT-style, keeps BN trainable)
model.qconfig = torch.ao.quantization.get_default_qat_qconfig(a.backend)
torch.ao.quantization.prepare_qat(model, inplace=True)             # 3) insert FakeQuantize modules
print('Fake-quant modules inserted:', sum('FakeQuantize' in type(m).__name__ for m in model.modules()))

# 4) fine-tune with fake quantization (observers and BN statistics frozen for the last epoch)
def hook(ep):
    if ep == a.epochs - 1 and a.epochs > 1:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
fit('qat', model, x, y, a.epochs, a.lr, wd=1e-4, schedule='cosine', epoch_hook=hook, max_steps_per_epoch=a.max_batches)

model.eval()
torch.save(model.state_dict(), a.out.replace('.pt', '_fakequant_state_dict.pth'))   # kept for QDQ-ONNX export
print(f'Fake-quantized (QAT) FP accuracy: {evaluate(model, xt, yt):.2f}%')
int8 = torch.ao.quantization.convert(model.cpu().eval(), inplace=False)   # 5) real INT8
acc = evaluate(int8, xt, yt)
print(f'QAT INT8 accuracy: {acc:.2f}%')
torch.jit.save(torch.jit.script(int8), a.out)                       # 6) export for inference
torch.save(int8.state_dict(), a.out.replace('.pt', '_state_dict.pth'))
print(f'Saved {a.out} ({file_mb(a.out):.2f} MB)')
