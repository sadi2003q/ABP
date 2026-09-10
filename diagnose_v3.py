"""
Run this in Colab:
    !python /content/ABP/diagnose_v3.py
(copy this file to /content/ABP/ first, or paste its contents into a cell)

Diagnoses why train_v3.py might be producing identical output to
train_v2.py despite being a different script.
"""
import sys
import os
import hashlib

REPO = "/content/ABP"
sys.path.insert(0, REPO)

print("=" * 70)
print("1. FILE IDENTITY CHECK")
print("=" * 70)

def md5(path):
    if not os.path.exists(path):
        return "MISSING"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

for fname in ["train_v2.py", "train_v3.py", "trainer_v2.py", "trainer_v3.py",
              "src/models/world_model_v2/model_v2.py",
              "src/models/world_model_v2/model_v3.py",
              "src/models/world_model_v2/pose_head_v2.py",
              "src/models/world_model_v2/pose_head_v3.py",
              "src/models/world_model_v2/__init__.py"]:
    path = os.path.join(REPO, fname)
    print(f"  {fname:55s} md5={md5(path)}  exists={os.path.exists(path)}")

print()
print("=" * 70)
print("2. IMPORT CHECK")
print("=" * 70)

try:
    from trainer_v3 import TrainerV3, TrainConfigV3
    print("  trainer_v3.TrainerV3 module   :", TrainerV3.__module__)
    print("  trainer_v3.TrainerV3 file     :", sys.modules[TrainerV3.__module__].__file__)
    print("  TrainerV3 MRO                 :", [c.__name__ for c in TrainerV3.__mro__])
except Exception as e:
    print("  FAILED to import trainer_v3:", repr(e))

try:
    from src.models.world_model_v2 import WorldModelV3
    print("  WorldModelV3 module           :", WorldModelV3.__module__)
    print("  WorldModelV3 file             :", sys.modules[WorldModelV3.__module__].__file__)
except Exception as e:
    print("  FAILED to import WorldModelV3:", repr(e))

print()
print("=" * 70)
print("3. _build_model SOURCE CHECK")
print("=" * 70)
import inspect
try:
    print(inspect.getsource(TrainerV3._build_model))
except Exception as e:
    print("  FAILED:", repr(e))

print()
print("=" * 70)
print("4. ACTUAL MODEL CLASS AFTER TrainerV3() CONSTRUCTION")
print("=" * 70)
try:
    cfg = TrainConfigV3(
        dataset_root="/content/drive/MyDrive/single_seq_root",
        sensors=("left_camera",),
        subset="imo",
        split="train",
        sequence=("scene15_dyn_test_06_000000",),
        batch_size=4,
        overfit_mode=True,
        use_ema=False,
        mixed_precision="none",
        num_workers=0,  # avoid Colab worker warnings for this quick check
    )
    t = TrainerV3(cfg)
    print("  type(t.model).__name__        :", type(t.model).__name__)
    print("  type(t.model).__module__      :", type(t.model).__module__)
    has_pose_head_v3 = "PoseHeadV3" in type(t.model.pose_head).__name__
    print("  pose_head class               :", type(t.model.pose_head).__name__)
    print("  model has imu_integration use :", hasattr(t.model, "_init_weights"))
except Exception as e:
    import traceback
    print("  FAILED to construct TrainerV3:")
    traceback.print_exc()

print()
print("=" * 70)
print("5. sys.path / duplicate ABP checks")
print("=" * 70)
for p in sys.path:
    if "ABP" in p or p == "":
        print("  sys.path entry:", repr(p))

print()
print("Done.")