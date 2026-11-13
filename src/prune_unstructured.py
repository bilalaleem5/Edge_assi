"""Unstructured magnitude pruning (iterative, global L1) + fine-tuning -> permanent sparse weights."""
import argparse, os, gzip, io, torch, torch.nn as nn
import torch.nn.utils.prune as prune
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--fp32', default=os.path.join(MODEL_DIR, 'resnet18_fp32.pth'))
ap.add_argument('--out', default=os.path.join(MODEL_DIR, 'resnet18_unstructured_pruned.pth'))
ap.add_argument('--target', type=float, default=0.80, help='final global sparsity of conv + fc weights')
ap.add_argument('--iters', type=int, default=5)
ap.add_argument('--ft-epochs', type=int, default=1, help='fine-tune epochs after EACH pruning iteration')
ap.add_argument('--final-epochs', type=int, default=1, help='extra fine-tune epochs after the last iteration')
a = ap.parse_args()

set_seed(); torch.set_num_threads(os.cpu_count())
x, y = load_split('train'); xt, yt = load_split('test')
model = ResNet18(); model.load_state_dict(torch.load(a.fp32))
base_acc = evaluate(model, xt, yt)
print(f'Original FP32 accuracy: {base_acc:.2f}%')

targets = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
total = sum(m.weight.numel() for m, _ in targets)

def sparsity():
    z = sum(int((m.weight == 0).sum()) for m, _ in targets)
    return 100.0 * z / total

# cubic sparsity schedule: s_i = target * (1 - (1 - i/n)^3); prune() amount is a fraction of the *remaining* weights
PROG = os.path.join(STATE_DIR, 'unstructured_progress.pt'); os.makedirs(STATE_DIR, exist_ok=True)
done = 0
if os.path.exists(PROG):                                    # resume support: restore model (incl. masks) after the last finished iteration
    pr = torch.load(PROG, weights_only=False); done = pr['done']
    if done > 0:
        for m, n in targets: prune.identity(m, n)
        model.load_state_dict(pr['sd'])
prev = a.target * (1 - (1 - done / a.iters) ** 3)
for it in range(done + 1, a.iters + 1):
    s_i = a.target * (1 - (1 - it / a.iters) ** 3)
    amount = 1 - (1 - s_i) / (1 - prev)
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)   # weight masking (weight_orig * weight_mask)
    prev = s_i
    print(f'[iter {it}/{a.iters}] target sparsity {100*s_i:.1f}% | measured {sparsity():.2f}%')
    ep = a.ft_epochs + (a.final_epochs if it == a.iters else 0)
    fit(f'unstructured_{it}', model, x, y, ep, 0.02, wd=1e-4, schedule='cosine')      # masks stay applied during fine-tuning
    print(f'          acc after fine-tune {evaluate(model, xt, yt):.2f}%')
    torch.save({'done': it, 'sd': model.state_dict()}, PROG)

for m, n in targets:
    prune.remove(m, n)          # make pruning permanent: fold the mask into the weight tensor
final_sp = sparsity()
acc = evaluate(model, xt, yt)
print(f'Permanent pruning done. Sparsity {final_sp:.2f}% | non-zero params {count_nonzero(model):,} / {count_params(model):,} | accuracy {acc:.2f}%')
torch.save(model.state_dict(), a.out)
buf = io.BytesIO(); torch.save(model.state_dict(), buf)
print(f'Saved {a.out} ({file_mb(a.out):.2f} MB dense, {len(gzip.compress(buf.getvalue()))/1e6:.2f} MB gzip-compressed)')
