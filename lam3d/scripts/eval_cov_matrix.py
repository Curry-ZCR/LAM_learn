#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pairwise PSNR / SSIM / LPIPS matrix among the four covariance methods.

There is no absolute photo ground truth (see docs/gt_protocol_feasibility.md),
so the three metrics the task asks for are reported as a 4x4 *agreement*
matrix over the frames already rendered by eval_cov_methods.py, plus a
changed-region restricted variant.  Agreement is NOT quality: it only says
how far each method moves the image relative to the others.

Reads /tmp/eval_cov_frames/{vec}_{method}.png (front view, SIZE=512).
"""
import json, math, itertools
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch

FRAMES = Path('/tmp/eval_cov_frames')
OUT = Path('/tmp/cov_matrix.json')
METHODS = ['lam', 'tga', 'rot_sync', 'vertex_then_polar']
PRESETS = ['smile', 'bigSmile', 'laugh', 'surprise', 'angry', 'sad', 'kiss', 'mouthOpen', 'blink']
VECTORS = ['zeros', 'ones', 'half', 'rand'] + PRESETS


def psnr(a, b, mask=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if mask is not None:
        if int(mask.sum()) < 50:
            return float('nan')
        a = a[mask]; b = b[mask]
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(255.0 ** 2 / mse))


def ssim(a, b, mask=None):
    ga = a.astype(np.float64).mean(-1); gb = b.astype(np.float64).mean(-1)
    if mask is not None:
        ys, xs = np.where(mask)
        if ys.size < 50:
            return float('nan')
        y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 9, ga.shape[0])
        x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 9, ga.shape[1])
        ga = ga[y0:y1, x0:x1]; gb = gb[y0:y1, x0:x1]
    if min(ga.shape) < 7:
        return float('nan')
    win = min(7, min(ga.shape) // 2 * 2 + 1)
    return float(structural_similarity(ga, gb, data_range=255.0, win_size=win))


class Lpips:
    def __init__(self):
        import lpips
        self.net = lpips.LPIPS(net='alex').cuda().eval()

    def __call__(self, a, b):
        ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].cuda()
        tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].cuda()
        with torch.no_grad():
            return float(self.net(ta, tb).item())


def main():
    net = Lpips()
    rep = {'vectors': [], 'methods': METHODS, 'pairwise': {}, 'consensus': {}}
    agg = {m: {'agree_psnr': [], 'agree_ssim': [], 'agree_lpips': []} for m in METHODS}
    for vec in VECTORS:
        imgs = {}
        ok = True
        for m in METHODS:
            f = FRAMES / ('%s_%s.png' % (vec, m))
            if not f.exists():
                ok = False
                break
            imgs[m] = np.asarray(Image.open(f).convert('RGB'))
        if not ok:
            continue
        rep['vectors'].append(vec)
        cells = {}
        for a, b in itertools.combinations(METHODS, 2):
            ia, ib = imgs[a], imgs[b]
            diff = np.abs(ia.astype(np.int16) - ib.astype(np.int16)).sum(-1) > 0
            cells[(a, b)] = {
                'psnr': psnr(ia, ib),
                'ssim': ssim(ia, ib),
                'lpips': net(ia, ib),
                'psnr_changed': psnr(ia, ib, diff) if diff.sum() >= 50 else float('nan'),
                'ssim_changed': ssim(ia, ib, diff) if diff.sum() >= 50 else float('nan'),
                'changed_pct': float(diff.mean() * 100.0),
            }
        rep['pairwise'][vec] = {'%s|%s' % k: v for k, v in cells.items()}
        cons = {}
        for m in METHODS:
            others = [cells[k] for k in cells if m in k]
            if not others:
                continue
            cons[m] = {
                'psnr': float(np.mean([o['psnr'] for o in others])),
                'ssim': float(np.mean([o['ssim'] for o in others])),
                'lpips': float(np.mean([o['lpips'] for o in others])),
            }
            agg[m]['agree_psnr'].append(cons[m]['psnr'])
            agg[m]['agree_ssim'].append(cons[m]['ssim'])
            agg[m]['agree_lpips'].append(cons[m]['lpips'])
        rep['consensus'][vec] = cons
        print('%-10s ' % vec + '  '.join(
            '%s:psnr %.1f ssim %.4f lpips %.3f' % (m[:6], cons[m]['psnr'], cons[m]['ssim'], cons[m]['lpips'])
            for m in METHODS if m in cons))
    rep['aggregate'] = {m: {k: float(np.mean(v)) for k, v in d.items() if v} for m, d in agg.items()}
    print()
    print('== aggregate agreement (mean over %d vectors, higher psnr/ssim = more central, lower lpips = more central)' % len(rep['vectors']))
    for m in METHODS:
        a = rep['aggregate'][m]
        print('  %-18s psnr %6.2f  ssim %.4f  lpips %.4f' % (m, a['agree_psnr'], a['agree_ssim'], a['agree_lpips']))
    OUT.write_text(json.dumps(rep, indent=1))
    print('wrote %s (%.1f KB)' % (OUT, OUT.stat().st_size / 1024.0))


if __name__ == '__main__':
    main()