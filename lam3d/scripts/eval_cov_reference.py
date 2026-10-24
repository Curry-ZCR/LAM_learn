#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fourth reference frame: the analytic deformation solution as pseudo-GT.

All three previously reported frames failed to rank the four methods:
  * vs the mesh pseudo-GT the systematic gap is ~11 dB and the
    method-varying part is only 1.0e-04 of it (see
    scripts/eval_cov_variance_decomp.py);
  * vs the LAM baseline the metric measures change, not quality;
  * there is no photo ground truth for this identity
    (docs/gt_protocol_feasibility.md).

This script adds a fourth frame that is independent of all four
implementations: the linearised deformation solution of GaussianAvatars /
SurFhead,

    Sigma_exact = J_v Sigma_0 J_v^T ,   Sigma_0 = R_0 diag(s_0^2) R_0^T

rendered by exact eigendecomposition back into the only channel the frozen
renderer reads (q, s).  It is an analytic solution of the deformation model,
NOT photometric truth, so the numbers say 'distance to the model solution',
not 'visual quality'.

Also reports how far the analytic solution itself leaves the contract box
(max_scale_ratio 1.35), which is why tga clamps and why rot_sync differs.

Run (builds a pipeline, ~120 s, must be backgrounded):
  cd /workspace/lam3d && nohup env PATH=/usr/local/miniconda3/envs/lam/bin:$PATH \
    PYTHONPATH=/workspace/lam3d:/workspace/LAM \
    /usr/local/miniconda3/envs/lam/bin/python scripts/eval_cov_reference.py \
    > /tmp/eval_ref.log 2>&1 &
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/workspace/lam3d')
sys.path.insert(0, '/workspace/LAM')
sys.path.insert(0, '/workspace/lam3d/scripts')

import torch  # noqa: E402

from lam3d.core import arkit_live, cov_switch, tga_covariance  # noqa: E402
from lam3d.core.pipeline import SingleImage3DPipeline  # noqa: E402
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat  # noqa: E402

import eval_cov_methods as E  # noqa: E402  reuse psnr/ssim/Lpips/render_full

IMAGE = Path('/workspace/LAM/assets/sample_input/messi.png')
JOB_DIR = Path('/tmp/eval_ref_job')
OUT_JSON = Path('/tmp/cov_reference.json')
OUT_DIR = Path('/tmp/eval_ref_frames')
METHODS = ['lam', 'tga', 'rot_sync', 'vertex_then_polar']
VECTORS = ['ones', 'half', 'rand', 'bigSmile', 'laugh', 'surprise', 'mouthOpen']
SIZE = 512


def log(msg):
    print('[ref] ' + msg, flush=True)


class ReferenceModulator(torch.nn.Module):
    """Writes Sigma_exact = J_v Sigma_0 J_v^T into (q, s) with no clamping."""

    def __init__(self, flame_model, q0, s0):
        super().__init__()
        self.base = tga_covariance._base_module().DynamicCovarianceModulator(flame_model)
        self.jac = tga_covariance.TgaCovarianceModulator(
            flame_model, jacobian_strength=1.0,
            max_scale_ratio=1.0e6, min_scale_ratio=1.0e-6)
        self.register_buffer('q0', q0.clone(), persistent=False)
        self.register_buffer('s0', s0.clone(), persistent=False)
        self.stat = {}

    def prepare_params(self, *a, **k):
        return self.base.prepare_params(*a, **k)

    def arkit_offsets(self, p):
        return self.base.arkit_offsets(p)

    def delta_rotation_scale(self, x0, xB):
        Jv = self.jac.vertex_jacobians(x0, xB)
        R0 = quat_to_matrix(self.q0)
        S0 = R0 @ torch.diag_embed(self.s0 * self.s0) @ R0.transpose(-1, -2)
        S = Jv @ S0.unsqueeze(0) if S0.dim() == 2 else Jv @ S0
        S = S @ Jv.transpose(-1, -2)
        S = 0.5 * (S + S.transpose(-1, -2))
        ev, vec = torch.linalg.eigh(S)
        ev = ev.clamp(min=0.0)
        s_ref = ev.sqrt()
        bad = torch.linalg.det(vec) < 0
        if bool(bad.any()):
            vec = vec.clone()
            vec[bad, :, -1] *= -1.0
        q_ref = matrix_to_quat(vec.transpose(-1, -2)).float()
        q_ref = q_ref / q_ref.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        ratio = (s_ref / self.s0.unsqueeze(0)).sort(-1).values
        self.stat = {
            'frac_gt_1p35': float((ratio[..., -1] > 1.35).float().mean()),
            'frac_gt_3': float((ratio[..., -1] > 3.0).float().mean()),
            'frac_gt_10': float((ratio[..., -1] > 10.0).float().mean()),
            'max_axis_ratio_max': float(ratio[..., -1].max()),
            'max_axis_ratio_p50': float(ratio[..., -1].median()),
        }
        return (None, (q_ref, s_ref))

    def delta_scale(self, payload, scaling):
        return payload

    def apply(self, delta_q, payload, rotation, scaling):
        q_ref, s_ref = payload
        return q_ref, s_ref


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = SingleImage3DPipeline(device='cuda', lazy_load=True)
    pipe.run(image_path=IMAGE, job_dir=JOB_DIR, job_id='evalref')
    sess = pipe.arkit_session
    renderer = sess['renderer']
    meta = arkit_live.arkit_meta()
    gs = sess['gs_model']
    q0 = gs.rotation.detach().float()
    s0 = gs.scaling.detach().float()
    log('ready %.1fs  N=%d' % (time.time() - t0, s0.shape[0]))

    ref = ReferenceModulator(renderer.flame_model, q0, s0).to('cuda').eval()
    net = E.Lpips()
    rep = {'vectors': [], 'methods': METHODS, 'per': {}, 'analytic_box': {}}

    for vec_name in VECTORS:
        vec = E.vector_for(meta, vec_name)
        # reference frame: analytic solution, no clamping
        renderer._cov_modulator = ref
        fr_ref = E.render_full(sess, vec, yaw=0.0, size=SIZE)
        box = dict(ref.stat)
        rep['analytic_box'][vec_name] = box
        gt = fr_ref['rgb']
        E_PIL = __import__('PIL.Image', fromlist=['Image'])
        E_PIL.fromarray(gt).save(OUT_DIR / ('%s_reference.png' % vec_name))
        log('%-10s analytic box: max-ratio p50 %.2f  >1.35 %.1f%%  >3 %.1f%%  >10 %.1f%%  max %.1f'
            % (vec_name, box['max_axis_ratio_p50'], 100 * box['frac_gt_1p35'],
               100 * box['frac_gt_3'], 100 * box['frac_gt_10'], box['max_axis_ratio_max']))

        entry = {}
        for name in METHODS:
            cov_switch.set_method(name, renderer=renderer)
            fr = E.render_full(sess, vec, yaw=0.0, size=SIZE)
            rgb = fr['rgb']
            E_PIL.fromarray(rgb).save(OUT_DIR / ('%s_%s.png' % (vec_name, name)))
            diff = (np.abs(rgb.astype(np.int16) - gt.astype(np.int16)).sum(-1) > 0)
            entry[name] = {
                'psnr': E.psnr(rgb, gt),
                'ssim': E.ssim(rgb, gt),
                'lpips': net(rgb, gt),
                'psnr_changed': (E.psnr(rgb, gt, diff) if diff.sum() >= 50 else None),
                'changed_pct': float(diff.mean() * 100.0),
            }
            log('  %-18s PSNR %6.2f  SSIM %.4f  LPIPS %.4f  changed %5.2f%%'
                % (name, entry[name]['psnr'], entry[name]['ssim'],
                   entry[name]['lpips'], entry[name]['changed_pct']))
        rep['vectors'].append(vec_name)
        rep['per'][vec_name] = entry

    keys = ['psnr', 'ssim', 'lpips']
    agg = {m: {k: float(np.mean([rep['per'][v][m][k] for v in rep['vectors']]))
               for k in keys} for m in METHODS}
    rep['aggregate'] = agg
    log('')
    log('== mean over %d vectors, distance to the analytic solution' % len(rep['vectors']))
    for m in METHODS:
        a = agg[m]
        log('  %-18s PSNR %6.2f  SSIM %.4f  LPIPS %.4f'
            % (m, a['psnr'], a['ssim'], a['lpips']))
    OUT_JSON.write_text(json.dumps(rep, indent=1))
    log('wrote %s  total %.1f s' % (OUT_JSON, time.time() - t0))


if __name__ == '__main__':
    main()