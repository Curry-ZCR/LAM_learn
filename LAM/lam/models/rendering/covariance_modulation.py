# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ARKit-blendshape driven dynamic covariance modulation for LAM's 3DGS head.

The canonical Gaussians produced by ``GSLayer`` are animated by
``GSRenderer.animate_gs_model`` through a single LBS skinning step
(``FlameHeadSubdivided.animation_forward``).  That step only moves the Gaussian
*means*; the predicted rotation quaternion ``q0`` and the activated scale ``s0``
are copied over unchanged, so during an expression the Gaussians slide across
the skin instead of rotating/stretching with it, which reads as stiffness or
tearing.

This module adds the missing shape change on top of the *existing* ARKit ->
FLAME machinery:

``f(B, x0, xB) -> (dq, ds)``

* ``B``  -- Apple ARKit blendshape vector, ``[B, 52]`` in ``[0, 1]``.
* ``x0`` -- posed mesh with the ARKit part neutral (``B = 0``).
* ``xB`` -- posed mesh driven by ``B`` (identical pose parameters).
* ``dq`` -- per-Gaussian rotation delta, Hamilton quaternion ``[B, N, 4]``.
* ``ds`` -- per-Gaussian additive scale delta, ``[B, N, 3]``.

How it works
------------
1.  The built-in ARKit basis shipped with LAM
    (``flame_arkit_bs.npy``, ``[52, 5023, 3]``, additive per-vertex offsets) is
    upsampled to the 20018-vertex Gaussian space with the *FLAME head's own*
    Loop subdividers (``FlameHeadSubdivided.upsample_mesh_cpu``) -- no
    hand-written mapping matrix is used, and the offsets are injected through
    ``FlameHead.animation_forward(dynamic_offset=...)`` so they are skinned
    exactly like FLAME's own expression blendshapes.
2.  For every upsampled triangle the in-plane deformation gradient is measured
    between the neutral pose ``x0`` and the driven pose ``xB`` using an
    orthonormal rest frame ``(t1, t2, n)``:

        A = Q P^-1        (P, Q = rest / deformed edge coordinates)

    ``A`` is the exact 2-D stretch+rotation of the triangle.
3.  ``A`` is split with a closed-form 2-D polar decomposition
    (``R = A (A^T A)^-1/2``, no SVD, no ``linalg`` call) into a rotation
    ``R`` and an area Jacobian ``det A``.  ``R`` is turned into a quaternion
    ``[cos(th/2), sin(th/2) n]`` -- a rotation about the face normal -- and
    averaged over the one-ring with *equal* weight per incident face (a unit
    quaternion is a direction, so that is the natural estimator).  ``log det A``
    is averaged over the same one-ring but *area* weighted, which is meaningful
    for a scalar area-Jacobian, into the local stretch signal ``u``.
4.  The per-Gaussian deltas are

        dq = normalize(q_ring)                         (exact ARAP rotation)
        ds = s0 * k * tanh(g * u)                      (soft-clipped stretch)

    and are composed with the predicted attributes in
    ``GSRenderer.animate_gs_model``:

        q' = normalize(dq (x) q0)                      (quaternion product)
        s' = s0 + ds                                   (activated scale)

Guarantees (checked by the self-review test-suite)
--------------------------------------------------
*   ``B = 0``  ->  ``x0 == xB`` bitwise, ``A == I`` bitwise, therefore
    ``dq == [1, 0, 0, 0]`` and ``ds == 0`` *exactly*; the renderer short
    circuits and the output is pixel-identical to the unmodified pipeline.
*   ``|B| <= 1`` ->  ``ds`` is bounded by ``s0 * k * tanh(g * |u|)`` with
    ``k < 1`` and a ``tanh`` soft clip, so ``s'`` stays in ``(0, 2 s0]`` and
    can never become NaN/Inf (no ``exp`` domain is involved: ``gs.scaling`` is
    the *activated* linear scale).
*   Everything is vectorised: only ``einsum``/``matmul``/``scatter_add`` are
    used, never a Python loop over the N Gaussians.
*   The composed 3-D covariance is never touched directly -- only the
    decoupled quaternion (re-normalised) and the scale vector are updated, and
    ``diff-gaussian-rasterization`` is used exactly as before.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ARKIT_BLENDSHAPE_NAMES",
    "ARKIT_PRESETS",
    "DynamicCovarianceModulator",
    "quaternion_multiply",
]

# Standard Apple ARKit blendshape ordering (alphabetical), 52 channels.  This is
# the ordering of ``flame_arkit_bs.npy`` (verified against the basis: the jaw
# and mouth channels carry by far the largest offsets).
ARKIT_BLENDSHAPE_NAMES = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
)

# Small library of ready-made expressions for the frontend panel (all values in
# [0, 1]; channels not listed stay 0).
ARKIT_PRESETS = {
    "neutral": {},
    "smile": {"mouthSmileLeft": 0.85, "mouthSmileRight": 0.85, "cheekSquintLeft": 0.25, "cheekSquintRight": 0.25},
    "bigSmile": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0, "mouthStretchLeft": 0.6, "mouthStretchRight": 0.6, "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4},
    "laugh": {"jawOpen": 0.7, "mouthSmileLeft": 0.9, "mouthSmileRight": 0.9, "mouthStretchLeft": 0.5, "mouthStretchRight": 0.5, "noseSneerLeft": 0.3, "noseSneerRight": 0.3},
    "surprise": {"jawOpen": 0.8, "eyeWideLeft": 0.9, "eyeWideRight": 0.9, "browInnerUp": 0.8, "browOuterUpLeft": 0.6, "browOuterUpRight": 0.6},
    "angry": {"browDownLeft": 0.9, "browDownRight": 0.9, "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6, "noseSneerLeft": 0.5, "noseSneerRight": 0.5, "jawForward": 0.3},
    "sad": {"browInnerUp": 0.9, "mouthFrownLeft": 0.8, "mouthFrownRight": 0.8, "mouthLowerDownLeft": 0.4, "mouthLowerDownRight": 0.4},
    "kiss": {"mouthPucker": 0.9, "mouthFunnel": 0.6, "mouthPressLeft": 0.3, "mouthPressRight": 0.3},
    "mouthOpen": {"jawOpen": 1.0, "mouthLowerDownLeft": 0.3, "mouthLowerDownRight": 0.3},
    "blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
    "winkLeft": {"eyeBlinkLeft": 1.0, "mouthSmileLeft": 0.5, "mouthSmileRight": 0.3},
    "winkRight": {"eyeBlinkRight": 1.0, "mouthSmileRight": 0.5, "mouthSmileLeft": 0.3},
    "cheekPuff": {"cheekPuff": 1.0, "jawForward": 0.2},
}

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

# Absolute path of the LAM checkout root.  This file lives at
# ``<LAM_ROOT>/lam/models/rendering/covariance_modulation.py``, so four
# ``dirname`` calls up from it land on the checkout root.  It is used only to
# make the *repo-relative* FLAME asset path (``flame_model.flame_model_dir`` ==
# ``./model_zoo/...``) resolve regardless of the process CWD -- the lam3d server
# is started from /workspace/lam3d, where ``./model_zoo/...`` does not exist.
_LAM_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _resolve_asset_path(path: str) -> str:
    """Resolve a possibly repo-relative FLAME asset path to an existing file.

    LAM locates its FLAME assets through ``flame_model_dir`` which is a
    *repo-relative* path (``./model_zoo/...``); it only resolves when the CWD is
    the LAM checkout root.  When ``path`` is relative and does not exist from the
    current CWD, retry it against the checkout root derived from this file's own
    location.  Absolute paths are returned untouched, and the rewritten
    candidate is only adopted when it actually exists, so a custom asset
    directory keeps working unchanged.
    """
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    candidate = os.path.join(_LAM_ROOT, os.path.normpath(path))
    return candidate if os.path.isfile(candidate) else path


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a (x) b`` for wxyz quaternions (pytorch3d order)."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


class DynamicCovarianceModulator(nn.Module):
    """Computes per-Gaussian rotation/scale deltas from ARKit blendshapes.

    Args:
        flame_model: the ``FlameHeadSubdivided`` instance owned by the
            renderer (used for its Loop subdividers and for ``faces_up``).
        arkit_bs_path: path of the built-in ARKit basis
            (``flame_arkit_bs.npy``, ``[52, 5023, 3]``).  Defaults to the file
            that ships next to ``flame2023.pkl`` inside LAM's model zoo.
        scale_strength: ``k`` -- upper bound of the relative scale change
            (``ds/s0``), applied through ``tanh``.
        stretch_gain: ``g`` -- slope of the ``tanh`` on the log area Jacobian.
        rot_strength: multiplier on the measured local rotation (1.0 = exact
            as-rigid-as-possible fit).
    """

    def __init__(
        self,
        flame_model: nn.Module,
        arkit_bs_path: Optional[str] = None,
        *,
        scale_strength: float = 0.35,
        stretch_gain: float = 2.0,
        rot_strength: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        # Hyperparameter contract, validated before anything expensive runs.
        # ``scale_strength`` is the *relative* bound of the additive scale
        # delta (``ds = s0 * k * tanh(...)``), so it must stay strictly inside
        # (0, 1): at ``k >= 1`` the additive delta can drive ``s' <= 0`` and
        # hand a negative variance to the rasterizer.
        if not 0.0 < scale_strength < 1.0:
            raise ValueError(
                "scale_strength must be in the open interval (0, 1) so that "
                "s' = s0 + ds stays strictly positive; got {!r}".format(scale_strength)
            )
        if not stretch_gain > 0:
            raise ValueError(
                "stretch_gain must be > 0 (slope of the tanh soft clip); "
                "got {!r}".format(stretch_gain)
            )
        if not eps > 0:
            raise ValueError(
                "eps must be > 0 (degeneracy floor of every normalisation); "
                "got {!r}".format(eps)
            )
        if arkit_bs_path is None:
            model_dir = getattr(flame_model, "flame_model_dir", None)
            if model_dir is None:
                raise ValueError(
                    "cannot locate the ARKit basis: pass arkit_bs_path explicitly"
                )
            arkit_bs_path = os.path.join(model_dir, "flame_arkit_bs.npy")
        # ``model_dir`` is repo-relative (``./model_zoo/...``); resolve it
        # against the LAM checkout root when the CWD is not that root (e.g. the
        # lam3d server, started from /workspace/lam3d).  Same file, just found.
        arkit_bs_path = _resolve_asset_path(arkit_bs_path)
        if not os.path.isfile(arkit_bs_path):
            raise FileNotFoundError(f"ARKit blendshape basis not found: {arkit_bs_path}")

        n_base = int(flame_model.v_template.shape[0])
        basis = np.load(arkit_bs_path).astype(np.float32)
        if basis.ndim != 3 or basis.shape[1] != n_base or basis.shape[2] != 3:
            raise ValueError(
                f"unexpected ARKit basis shape {tuple(basis.shape)}; expected "
                f"(52, {n_base}, 3)"
            )
        n_arkit = int(basis.shape[0])

        # Upsample the 5023-vertex ARKit basis to the Gaussian (subdivided)
        # vertex space with the FLAME head's own subdividers -- the very same
        # operator that produces ``v_template_up`` / ``shapedirs_up``.
        feats = [torch.from_numpy(basis[k]) for k in range(n_arkit)]
        with torch.no_grad():
            ret = flame_model.upsample_mesh_cpu(
                flame_model.v_template.detach().float().cpu(), feats
            )
        arkit_bs_up = torch.stack([r.detach().float() for r in ret[1:]], dim=0)

        faces = flame_model.faces_up.detach().long().cpu()
        if int(faces.max()) + 1 != arkit_bs_up.shape[1]:
            raise ValueError(
                "ARKit basis upsampling mismatch: "
                f"{int(faces.max()) + 1} faces vertices vs {arkit_bs_up.shape[1]} basis vertices"
            )

        self.register_buffer("arkit_bs_up", arkit_bs_up, persistent=False)
        self.register_buffer("faces", faces, persistent=False)
        # destination vertex id of every (face, corner) pair -- scatter index
        self.register_buffer("faces_flat", faces.reshape(-1), persistent=False)
        self.n_arkit = n_arkit
        self.scale_strength = float(scale_strength)
        self.stretch_gain = float(stretch_gain)
        self.rot_strength = float(rot_strength)
        self.eps = float(eps)

    # ------------------------------------------------------------------ utils
    @property
    def num_gaussians(self) -> int:
        return int(self.arkit_bs_up.shape[1])

    @staticmethod
    def is_neutral(arkit_params: Optional[torch.Tensor]) -> bool:
        """True when every ARKit channel is exactly zero (static pass-through)."""
        if arkit_params is None:
            return True
        return bool(arkit_params.abs().sum() == 0)

    def prepare_params(self, arkit_params, num_view: int, device, dtype) -> torch.Tensor:
        """Validate/broadcast an ARKit vector to ``[num_view, 52]``."""
        if not torch.is_tensor(arkit_params):
            arkit_params = torch.as_tensor(arkit_params)
        p = arkit_params.to(device=device, dtype=dtype).reshape(-1, self.n_arkit)
        if p.shape[0] == 1 and num_view != 1:
            p = p.expand(num_view, -1)
        if p.shape[0] != num_view:
            raise ValueError(
                f"arkit_params batch {p.shape[0]} does not match num_view {num_view}"
            )
        return p.contiguous()

    def arkit_offsets(self, arkit_params: torch.Tensor) -> torch.Tensor:
        """Per-Gaussian canonical offsets ``[num_view, N, 3]`` for ``B``."""
        # single GEMM instead of a python loop over the 52 channels
        flat = self.arkit_bs_up.reshape(self.n_arkit, -1)          # [52, N*3]
        return torch.matmul(arkit_params, flat).view(
            arkit_params.shape[0], self.arkit_bs_up.shape[1], 3
        )

    # ------------------------------------------------------------------- core
    def delta_rotation_scale(self, x0: torch.Tensor, xB: torch.Tensor):
        """Measure the local deformation between two posed meshes.

        Args:
            x0: ``[B, N, 3]`` posed mesh, ARKit neutral.
            xB: ``[B, N, 3]`` posed mesh driven by the ARKit vector (same pose
                parameters as ``x0``).
        Returns:
            ``(delta_q, stretch)`` with ``delta_q`` ``[B, N, 4]`` Hamilton
            quaternions and ``stretch`` ``[B, N, 1]`` log area Jacobians.
        """
        faces = self.faces.to(x0.device)
        faces_flat = self.faces_flat.to(x0.device)
        n_vert = x0.shape[-2]
        batch = x0.shape[0]

        a, b, c = faces.unbind(dim=-1)                                  # [F]
        e1 = x0[..., b, :] - x0[..., a, :]                              # [B,F,3]
        e2 = x0[..., c, :] - x0[..., a, :]
        d1 = xB[..., b, :] - xB[..., a, :]
        d2 = xB[..., c, :] - xB[..., a, :]

        # orthonormal in-plane frame of the rest edge pair
        normal = torch.cross(e1, e2, dim=-1)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        t1 = e1 / e1.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        t2 = torch.cross(normal, t1, dim=-1)                            # unit

        def _coords(vec):
            return torch.stack(
                ((vec * t1).sum(dim=-1), (vec * t2).sum(dim=-1)), dim=-1
            )                                                           # [B,F,2]

        p00, p01 = _coords(e1).unbind(dim=-1)
        p10, p11 = _coords(e2).unbind(dim=-1)
        q00, q01 = _coords(d1).unbind(dim=-1)
        q10, q11 = _coords(d2).unbind(dim=-1)

        det_p = (p00 * p11 - p01 * p10).clamp(min=self.eps)             # 2*area
        det_q = q00 * q11 - q01 * q10

        # A = Q P^-1  (in-plane deformation gradient, orthonormal basis)
        a00 = (q00 * p11 - q01 * p10) / det_p
        a01 = (q01 * p00 - q00 * p01) / det_p
        a10 = (q10 * p11 - q11 * p10) / det_p
        a11 = (q11 * p00 - q10 * p01) / det_p

        # exact 2-D polar factor R = A (A^T A)^(-1/2)  (see _polar_rotation_2d)
        r00, r01, r10, r11 = self._polar_rotation_2d(a00, a01, a10, a11)

        # rotation angle about the rest normal -> face quaternion (w >= 0, so
        # one-ring averaging needs no sign fix)
        theta = torch.atan2(r10, r00) * self.rot_strength
        half = 0.5 * theta
        sin_half = torch.sin(half)
        face_quat = torch.stack(
            (torch.cos(half), sin_half * normal[..., 0], sin_half * normal[..., 1], sin_half * normal[..., 2]),
            dim=-1,
        )                                                               # [B,F,4]

        # local stretch: log of the area Jacobian, soft domain-checked.
        # Determinants re-evaluated in float64: for sliver faces (det_p down
        # to ~1e-8 on the real mesh) the float32 subtraction cancels several
        # digits and injects a spurious stretch signal.
        det_p64 = p00.double() * p11.double() - p01.double() * p10.double()
        det_q64 = q00.double() * q11.double() - q01.double() * q10.double()
        log_jac = torch.log((det_q64 / det_p64).clamp(min=1e-4, max=1e4)).to(x0.dtype)

        # ---- one-ring averages, fully vectorised ---------------------------
        # The quaternion average is deliberately UNWEIGHTED (a plain
        # ``scatter_add_`` of ``face_quat``): a unit quaternion is a direction,
        # so equal weight per incident face is the natural estimator, and the
        # ``w = cos(half) >= 0`` convention above keeps the sum well defined
        # without any sign fix.  Area weighting is only meaningful for the
        # scalar area-Jacobian, which is where it is applied.
        idx = faces_flat.view(1, -1, 1).expand(batch, -1, 4)
        quat_sum = torch.zeros(batch, n_vert, 4, device=x0.device, dtype=x0.dtype)
        quat_sum.scatter_add_(1, idx, face_quat.repeat_interleave(3, dim=1))
        quat_norm = quat_sum.norm(dim=-1, keepdim=True)
        delta_q = quat_sum / quat_norm.clamp(min=1e-12)
        degenerate = quat_norm < 1e-8
        if bool(degenerate.any()):
            ident = torch.tensor(_IDENTITY_QUAT, device=x0.device, dtype=x0.dtype)
            delta_q = torch.where(degenerate, ident.expand_as(delta_q), delta_q)

        area = 0.5 * det_p                                              # [B,F]
        idx1 = faces_flat.view(1, -1).expand(batch, -1)
        area3 = area.repeat_interleave(3, dim=1)                        # [B,3F]
        jac3 = log_jac.repeat_interleave(3, dim=1)
        acc = torch.zeros(batch, n_vert, device=x0.device, dtype=x0.dtype)
        acc.scatter_add_(1, idx1, jac3 * area3)
        cnt = torch.zeros(batch, n_vert, device=x0.device, dtype=x0.dtype)
        cnt.scatter_add_(1, idx1, area3)
        stretch = acc / cnt.clamp(min=self.eps)                         # [B,N]

        return delta_q, stretch.unsqueeze(-1)

    def _polar_rotation_2d(self, a00, a01, a10, a11):
        """Exact 2-D polar factor R = A (A^T A)^(-1/2) for a batch of 2x2.

        Returns (r00, r01, r10, r11), each with the broadcast shape of the
        inputs.  Written as the matrix product A . G^(-1/2) -- the cross
        terms are NOT optional: dropping them (a Hadamard product) silently
        destroys the rotation signal.
        """
        # G = A^T A, symmetric positive semi-definite by construction.
        g00 = a00 * a00 + a10 * a10
        g01 = a00 * a01 + a10 * a11
        g11 = a01 * a01 + a11 * a11
        det_g = (g00 * g11 - g01 * g01).clamp(min=self.eps)
        sqrt_det_g = det_g.sqrt()
        det_m = ((g00 + sqrt_det_g) * (g11 + sqrt_det_g) - g01 * g01).clamp(min=self.eps)
        scale = (g00 + g11 + 2.0 * sqrt_det_g).clamp(min=self.eps).sqrt()
        # G^(-1/2) = scale/det_m * [[g11+s, -g01], [-g01, g00+s]]  with
        # s = sqrt(det G); R = A . G^(-1/2) is a genuine matrix product, so
        # every entry mixes a row of A with a column of G^(-1/2).
        r00 = scale * (a00 * (g11 + sqrt_det_g) - a01 * g01) / det_m
        r01 = scale * (a01 * (g00 + sqrt_det_g) - a00 * g01) / det_m
        r10 = scale * (a10 * (g11 + sqrt_det_g) - a11 * g01) / det_m
        r11 = scale * (a11 * (g00 + sqrt_det_g) - a10 * g01) / det_m
        return r00, r01, r10, r11

    def delta_scale(self, stretch: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
        """``ds = s0 * k * tanh(g * u)`` -- additive, bounded, positive s'."""
        gain = self.scale_strength * torch.tanh(self.stretch_gain * stretch)
        return scaling.unsqueeze(0) * gain

    def apply(self, delta_q: torch.Tensor, delta_scale: torch.Tensor,
              rotation: torch.Tensor, scaling: torch.Tensor):
        """Compose the deltas with the predicted Gaussian attributes.

        Args:
            delta_q: ``[B, N, 4]``; delta_scale: ``[B, N, 3]``;
            rotation/scaling: ``[N, 4]`` / ``[N, 3]`` canonical attributes.
        Returns:
            ``(rotation', scaling')`` with shapes ``[B, N, 4]`` / ``[B, N, 3]``.
        """
        q = quaternion_multiply(delta_q, rotation.unsqueeze(0).to(delta_q.dtype))
        q = F.normalize(q, dim=-1)
        s = scaling.unsqueeze(0).to(delta_scale.dtype) + delta_scale
        # Defensive floor: ``delta_scale`` is additive and bounded by
        # ``scale_strength * s0`` (validated in ``__init__`` to be < s0), so
        # this clamp is a no-op for every legal configuration -- it only guards
        # a caller that bypasses that contract.  At ARKit == 0 the delta is
        # bitwise 0, so ``s`` stays bit-identical to ``s0``.
        s = s.clamp(min=self.eps)
        return q, s
