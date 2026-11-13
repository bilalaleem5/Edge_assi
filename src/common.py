"""Shared utilities: data loading, quantization-ready ResNet-18, train / eval helpers."""
import os, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.nn.quantized import FloatFunctional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODEL_DIR = os.path.join(ROOT, 'models')
RESULT_DIR = os.path.join(ROOT, 'results')
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
WIDTH = 16          # base channel width (standard ResNet-18 uses 64); see README
SEED = 42


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_split(split):
    d = np.load(os.path.join(DATA_DIR, f'cifar10_{split}.npz'))
    x = torch.from_numpy(d['x']).permute(0, 3, 1, 2).contiguous()   # uint8 N,3,32,32
    y = torch.from_numpy(d['y'])
    return x, y


def normalize(xb):
    return (xb.float() / 255.0 - MEAN) / STD


def augment(xb):
    """Random crop (pad 4) + horizontal flip on a uint8 batch."""
    n = xb.size(0)
    xp = F.pad(xb, (4, 4, 4, 4))
    out = torch.empty_like(xb)
    i = torch.randint(0, 9, (n,)); j = torch.randint(0, 9, (n,)); fl = torch.rand(n) < 0.5
    for k in range(n):
        im = xp[k, :, i[k]:i[k] + 32, j[k]:j[k] + 32]
        out[k] = im.flip(2) if fl[k] else im
    return out


def batches(x, y, bs, train=False):
    idx = torch.randperm(x.size(0)) if train else torch.arange(x.size(0))
    for s in range(0, x.size(0), bs):
        b = idx[s:s + bs]
        xb = x[b]
        if train:
            xb = augment(xb)
        yield normalize(xb), y[b]


# ------------------------- Quantization-ready ResNet-18 -------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, mid=None):
        super().__init__()
        mid = mid or cout           # 'mid' lets structured pruning shrink the inner channels
        self.conv1 = nn.Conv2d(cin, mid, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.add_relu = FloatFunctional()    # quantization-friendly residual add + ReLU

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        idt = x if self.downsample is None else self.downsample(x)
        return self.add_relu.add_relu(out, idt)


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18: 3x3 stem, [2,2,2,2] BasicBlocks, 4 stages (widths w,2w,4w,8w)."""
    def __init__(self, num_classes=10, width=WIDTH, mids=None):
        super().__init__()
        w = width
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.relu = nn.ReLU()
        cfg = [(w, w, 1), (w, w, 1), (w, 2*w, 2), (2*w, 2*w, 1),
               (2*w, 4*w, 2), (4*w, 4*w, 1), (4*w, 8*w, 2), (8*w, 8*w, 1)]
        blocks = []
        for n, (ci, co, s) in enumerate(cfg):
            blocks.append(BasicBlock(ci, co, s, mid=None if mids is None else mids[n]))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 * w, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = torch.flatten(self.pool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def fuse_model(self):
        """Fuse Conv+BN(+ReLU) so they quantize as single ops. Works in eval mode (PTQ) or train mode (QAT)."""
        fn = fuse_modules if not self.training else torch.ao.quantization.fuse_modules_qat
        fn(self, [['conv1', 'bn1', 'relu']], inplace=True)
        for b in self.blocks:
            fn(b, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)
            if b.downsample is not None:
                fn(b.downsample, [['0', '1']], inplace=True)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_nonzero(m):
    return sum(int((p != 0).sum()) for p in m.parameters())


def file_mb(path):
    return os.path.getsize(path) / 1e6


# ------------------------- train / eval -------------------------
@torch.no_grad()
def evaluate(model, x, y, bs=500):
    model.eval()
    correct = 0
    for xb, yb in batches(x, y, bs):
        correct += (model(xb).argmax(1) == yb).sum().item()
    return 100.0 * correct / x.size(0)


START = time.time()
BUDGET = float(os.environ.get('BUDGET', '1e9'))   # seconds per process; training saves its state and exits (code 75) when exceeded
STATE_DIR = os.path.join(ROOT, 'state')


def fit(tag, model, x, y, epochs, lr, bs=128, wd=5e-4, schedule='onecycle', xt=None, yt=None,
        epoch_hook=None, max_steps_per_epoch=None, log=print):
    """SGD training loop that is resumable: its state (model/optimizer/scheduler/step) is checkpointed under state/<tag>.pt,
    so an interrupted run simply continues where it stopped when the script is started again."""
    import sys
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f'{tag}.pt')
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    nb = (x.size(0) + bs - 1) // bs
    if max_steps_per_epoch:
        nb = min(nb, max_steps_per_epoch)
    total = epochs * nb
    if schedule == 'onecycle':
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.25)
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    step, run = 0, [0.0, 0, 0]
    if os.path.exists(path):
        st = torch.load(path, weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sch.load_state_dict(st['sched'])
        step, run = st['step'], st['run']
        log(f'  [resume {tag}] continuing from step {step}/{total}')

    def save():
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sch.state_dict(), 'step': step, 'run': run}, path + '.tmp')
        os.replace(path + '.tmp', path)

    last_ep = -1
    while step < total:
        ep, j = divmod(step, nb)
        if ep != last_ep:
            if epoch_hook: epoch_hook(ep)
            last_ep = ep
        model.train()
        perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(1000 + ep))
        b = perm[j * bs:(j + 1) * bs]
        xb, yb = normalize(augment(x[b])), y[b]
        opt.zero_grad(set_to_none=True)
        out = model(xb); loss = F.cross_entropy(out, yb); loss.backward(); opt.step(); sch.step()
        run[0] += loss.item() * xb.size(0); run[1] += (out.argmax(1) == yb).sum().item(); run[2] += xb.size(0)
        step += 1
        if step % nb == 0:
            msg = f'  epoch {step//nb}/{epochs} loss {run[0]/run[2]:.4f} train-acc {100*run[1]/run[2]:.2f}%'
            if xt is not None: msg += f' | test-acc {evaluate(model, xt, yt):.2f}%'
            log(msg); run = [0.0, 0, 0]; save()
        elif step % 60 == 0:
            save()
        if time.time() - START > BUDGET and step < total:
            save(); log(f'  [budget reached at step {step}/{total}; state saved - re-run the same command to continue]'); sys.exit(75)
    save()
    return model


def count_macs(model, size=(1, 3, 32, 32)):
    """Multiply-accumulate count for Conv2d / Linear layers of a float model (one forward pass)."""
    macs = [0]
    def conv_hook(m, i, o):
        macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def fc_hook(m, i, o):
        macs[0] += m.in_features * m.out_features
    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d): hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear): hs.append(m.register_forward_hook(fc_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(size))
    for h in hs: h.remove()
    return macs[0]
