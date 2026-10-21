"""Probe: which FLAME head configuration yields the same vertex count as the LAM canonical GS points?"""
import sys
sys.path.insert(0, '/workspace/LAM')
import numpy as np
import torch

from lam.models.rendering.flame_model.flame import FlameHead, FlameHeadSubdivided

COMMON = dict(
    shape_params=300,
    expr_params=100,
    add_shoulder=False,
    flame_model_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl',
    flame_lmk_embedding_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy',
    flame_template_mesh_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj',
    flame_parts_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl',
)

for subdiv in (0, 1, 2):
    for teeth in (False, True):
        try:
            m = FlameHeadSubdivided(subdivide_num=subdiv, add_teeth=teeth, **COMMON)
        except Exception as e:  # noqa: BLE001
            print(f'subdiv={subdiv} teeth={teeth} FAILED: {type(e).__name__}: {e}')
            continue
        v = m.v_template_up.shape[0]
        f = m.faces_up.shape[0]
        with torch.no_grad():
            sp = torch.zeros(1, 300)
            cano = m.get_cano_verts(sp)
        print(f'subdiv={subdiv} teeth={teeth}: v_template_up={v} faces_up={f} get_cano_verts={tuple(cano.shape)}')
