"""Train the FP32 baseline ResNet-18 on CIFAR-10 and save the checkpoint."""
import argparse, os, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=10)
ap.add_argument('--lr', type=float, default=0.1)
ap.add_argument('--out', default=os.path.join(MODEL_DIR, 'resnet18_fp32.pth'))
a = ap.parse_args()

set_seed(); torch.set_num_threads(os.cpu_count())
os.makedirs(MODEL_DIR, exist_ok=True)
x, y = load_split('train'); xt, yt = load_split('test')
model = ResNet18()
print(f'ResNet-18 (width={WIDTH}) params: {count_params(model):,}')
fit('baseline', model, x, y, a.epochs, a.lr, xt=xt, yt=yt)
acc = evaluate(model, xt, yt)
torch.save(model.state_dict(), a.out)
print(f'FP32 test accuracy: {acc:.2f}%  -> saved {a.out} ({file_mb(a.out):.2f} MB)')
