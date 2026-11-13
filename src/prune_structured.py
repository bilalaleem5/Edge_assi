"""Structured channel pruning of ResNet-18 + physical architecture reconstruction + fine-tuning.
Residual connections force every block's output to keep the same width, so we prune the *internal* channels of each
BasicBlock (output filters of conv1 == input channels of conv2). Filters are ranked by L1 norm."""
import argparse, os, json, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--fp32', default=os.path.join(MODEL_DIR, 'resnet18_fp32.pth'))
ap.add_argument('--out', default=os.path.join(MODEL_DIR, 'resnet18_structured_pruned.pth'))
ap.add_argument('--ratio', type=float, default=0.5, help='fraction of internal channels removed in every block')
ap.add_argument('--ft-epochs', type=int, default=3)
a = ap.parse_args()

set_seed(); torch.set_num_threads(os.cpu_count())
x, y = load_split('train'); xt, yt = load_split('test')
orig = ResNet18(); orig.load_state_dict(torch.load(a.fp32)); orig.eval()
print(f'Original: {evaluate(orig, xt, yt):.2f}% | params {count_params(orig):,} | MACs {count_macs(orig)/1e6:.2f}M')

# 1) identify the least important channels (L1 norm of each conv1 filter)
keeps = []
for b in orig.blocks:
    imp = b.conv1.weight.detach().abs().sum(dim=(1, 2, 3))
    k = max(1, int(round(imp.numel() * (1 - a.ratio))))
    keeps.append(torch.sort(torch.topk(imp, k).indices).values)
mids = [len(k) for k in keeps]
print('Channels kept per block (mid width):', mids)

# 2) reconstruct a physically smaller network and copy surviving weights (no masked zeros remain)
new = ResNet18(mids=mids)
sd = orig.state_dict(); nsd = new.state_dict()
for name in nsd:
    nsd[name] = sd[name].clone() if sd[name].shape == nsd[name].shape else None
for i, (bo, bn, keep) in enumerate(zip(orig.blocks, new.blocks, keeps)):
    p = f'blocks.{i}.'
    nsd[p + 'conv1.weight'] = bo.conv1.weight.detach()[keep].clone()
    for t in ['weight', 'bias', 'running_mean', 'running_var']:
        nsd[p + 'bn1.' + t] = getattr(bo.bn1, t).detach()[keep].clone()
    nsd[p + 'conv2.weight'] = bo.conv2.weight.detach()[:, keep].clone()
assert all(v is not None for v in nsd.values())
new.load_state_dict(nsd); new.eval()
print(f'Pruned (no fine-tune): {evaluate(new, xt, yt):.2f}% | params {count_params(new):,} | MACs {count_macs(new)/1e6:.2f}M')

# 3) fine-tune to recover accuracy
fit('structured', new, x, y, a.ft_epochs, 0.03, wd=5e-4, xt=xt, yt=yt)
acc = evaluate(new, xt, yt)
print(f'Structured-pruned + fine-tuned: {acc:.2f}% | params {count_params(new):,} ({100*count_params(new)/count_params(orig):.1f}% of original) | MACs {count_macs(new)/1e6:.2f}M')
torch.save({'mids': mids, 'state_dict': new.state_dict()}, a.out)
print(f'Saved {a.out} ({file_mb(a.out):.2f} MB)')
