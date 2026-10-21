"""Verify: can we attach FLAME's subdivided topology (39904 faces, 20018 verts)
to LAM's canonical Gaussian positions (framework_img.obj) and get a sane mesh?"""
import sys
sys.path.insert(0, '/workspace/LAM')
import numpy as np
import trimesh
import torch
from plyfile import PlyData

from lam.models.rendering.flame_model.flame import FlameHeadSubdivided

# --- 1. LAM canonical geometry (20018 pts + rgb), saved by lam.py as framework_img.obj
verts, cols = [], []
for line in open('/workspace/LAM/exps/cano_gs/framework_img.obj'):
    if line.startswith('v '):
        p = line.split()
        verts.append([float(p[1]), float(p[2]), float(p[3])])
        cols.append([float(p[4]) if len(p) > 4 else 1.0,
                     float(p[5]) if len(p) > 5 else 1.0,
                     float(p[6]) if len(p) > 6 else 1.0])
verts = np.array(verts, dtype=np.float64)
cols = np.array(cols, dtype=np.float64)
print('framework_img.obj verts', verts.shape, 'color range', cols.min().round(3), cols.max().round(3))

# --- 2. offset ply saved with offset2xyz=True
d = PlyData.read('/workspace/LAM/exps/cano_gs/status_gs_offset.ply')
v = d['vertex']
off = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], 1).astype(np.float64)
print('offset ply verts', off.shape)

# --- 3. FLAME topology
flame = FlameHeadSubdivided(
    shape_params=300, expr_params=100, add_teeth=False, add_shoulder=False,
    flame_model_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl',
    flame_lmk_embedding_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy',
    flame_template_mesh_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj',
    flame_parts_path='/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl',
    subdivide_num=1,
)
faces_up = flame.faces_up.cpu().numpy()
v_template_up = flame.v_template_up.cpu().numpy().astype(np.float64)
print('faces_up', faces_up.shape, 'max idx', faces_up.max(), 'expected verts', v_template_up.shape)

assert verts.shape[0] == v_template_up.shape[0] == 20018, 'vertex count mismatch'
assert faces_up.max() == 20017, 'face index out of range'

# --- 4. does framework_img.obj == FLAME(shape) + offset?  shaped = verts - off
shaped = verts - off
# compare with FLAME neutral template (shape=0) -> should differ only by shape blend
delta = shaped - v_template_up
print('shaped - v_template_neutral : mean %.5f  max %.5f (FLAME 形状混合量)' % (
    np.abs(delta).mean(), np.abs(delta).max()))

# --- 5. build the actual 3D model mesh
mesh = trimesh.Trimesh(vertices=verts, faces=faces_up, process=False)
print('MESH: v %s f %s' % (mesh.vertices.shape, mesh.faces.shape))
print('  watertight=%s  euler=%s  volume=%.6f' % (mesh.is_watertight, mesh.euler_number, mesh.volume))
print('  bbox', verts.min(0).round(4), verts.max(0).round(4))
comps = mesh.split(only_watertight=False)
print('  components:', len(comps), 'sizes', sorted([len(c.faces) for c in comps], reverse=True)[:6])
mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8))
mesh.export('/tmp/lam3d_mesh_test.glb')
mesh.export('/tmp/lam3d_mesh_test.obj')
import os
print('  glb %.1f KB   obj %.1f KB' % (os.path.getsize('/tmp/lam3d_mesh_test.glb') / 1024,
                                       os.path.getsize('/tmp/lam3d_mesh_test.obj') / 1024))

# --- 6. reload via trimesh (three.js GLTFLoader sanity proxy)
back = trimesh.load('/tmp/lam3d_mesh_test.glb', process=False)
print('reloaded glb:', type(back).__name__, back.vertices.shape, back.faces.shape,
      'has_colors', back.visual.kind)
