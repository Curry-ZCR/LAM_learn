
import sys, inspect
sys.path.insert(0,"/workspace/lam3d")
from lam3d.core.rot_sync import RotSyncModulator
src=inspect.getsource(RotSyncModulator)
i=src.find("def rotation_field"); print(src[i:i+4000])
