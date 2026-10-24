#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Variance decomposition of the pixel metrics against the mesh pseudo-GT.

For each vector the four methods render the SAME geometry with slightly
different covariances.  Splitting the total squared error against the mesh
pseudo-GT into a systematic part (shared by all methods) and a
method-varying part answers one question quantitatively:

    is the 11 dB systematic gap so large that no PSNR/SSIM ranking of the
    four methods can be meaningful?

E_sys  = ||mean_m(I_m) - I_gt||^2      (what every method gets wrong equally)
E_var  = mean_m ||I_m - mean_m(I_m)||^2 (what the methods disagree about)
ratio  = E_var / E_sys                 (the only part a ranking could use)

Reads /tmp/eval_cov_frames/{vec}_{gt,method}.png, writes /tmp/cov_variance.json
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

FRAMES = Path('/tmp/eval_cov_frames')
OUT = Path('/tmp/cov_variance.json')
METHODS = ['lam', 'tga', 'rot_sync', 'vertex_then_polar']
VECTORS = ['zeros', 'ones', 'half', 'rand', 'smile', 'bigSmile', 'laugh',
           'surprise', 'angry', 'sad', 'kiss', 'mouthOpen', 'blink']


def load(p):
    return np.asarray(Image.open(p).convert('RGB'), np.float64)


def main():
    rep = {}
    for vec in VECTORS:
        gt_p = FRAMES / ('%s_gt.png' % vec)
        if not gt_p.exists():
            continue
        gt = load(gt_p)
        ms = []
        for m in METHODS:
            f = FRAMES / ('%s_%s.png' % (vec, m))
            if not f.exists():
                ms = []
                break
            ms.append(load(f))
        if not ms:
            continue
        stack = np.stack(ms, 0)
        mean = stack.mean(0)
        sys_e = float(np.mean((mean - gt) ** 2))
        var_e = float(np.mean(stack ** 2) - np.mean(mean ** 2))
        if var_e < 0:
            var_e = float(np.mean((stack - mean[None]) ** 2))
        per = {m: float(np.mean((x - gt) ** 2)) for m, x in zip(METHODS, ms)}
        rep[vec] = {
            'mse_systematic': sys_e,
            'mse_method_varying': var_e,
            'ratio_var_over_sys': var_e / max(sys_e, 1e-20),
            'mse_per_method': per,
            'spread_max_min': float(max(per.values()) - min(per.values())),
            'shared_by_all_frac': float(1.0 - var_e / max(np.mean(list(per.values())), 1e-20)),
        }
        print('%-10s E_sys %8.2f  E_var %8.3f  ratio %.2e  MSE spread %.2e  shared %.3f'
              % (vec, sys_e, var_e, rep[vec]['ratio_var_over_sys'],
                 rep[vec]['spread_max_min'], rep[vec]['shared_by_all_frac']))
    if rep:
        keys = ['mse_systematic', 'mse_method_varying', 'ratio_var_over_sys', 'spread_max_min']
        agg = {k: float(np.mean([v[k] for v in rep.values()])) for k in keys}
        rep['_aggregate'] = agg
        print()
        print('mean over %d vectors: E_sys %.2f  E_var %.3f  ratio %.2e  spread %.2e'
              % (len(rep) - 1, agg['mse_systematic'], agg['mse_method_varying'],
                 agg['ratio_var_over_sys'], agg['spread_max_min']))
        OUT.write_text(json.dumps(rep, indent=1))
        print('wrote %s' % OUT)


if __name__ == '__main__':
    main()