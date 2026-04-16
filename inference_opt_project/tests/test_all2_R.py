import numpy as np
import time
import descriptor_all_R
import descriptor_all2_R
from cul_f2_r_utils import load_model

model = load_model("models/linear_model_R.npz")
all_ls = model["all_ls"]
radial_pairs = model["RADIAL_PAIRS"]
r_c = model["CUTOFF"]

n_atoms = 2
nrad = len(radial_pairs)
n_base_feat = len(all_ls)
# total valid features depends on all_ls and radial_pairs,
# descriptor_all_R returns an array with shape (N, n_feat)

pos = np.array([
    [0.0, 0.0, 0.0],
    [1.5, 1.5, 1.5],
], dtype=np.float64)

lat = np.array([
    [3.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 0.0, 3.0],
], dtype=np.float64)

# Run baseline first to get n_feat
Xe, Xf, Xs = descriptor_all_R.compute_all_descriptors(
    pos, lat, 1.0, all_ls, radial_pairs, r_c, True
)

n_feat = Xe.shape[1]
print(f"n_feat from descriptor_all_R: {n_feat}")

# Create dummy coef matching n_feat
np.random.seed(42)
dummy_coef = np.random.randn(n_feat)

# Baseline projection
e_baseline = Xe @ dummy_coef
f_baseline = Xf @ dummy_coef

# Run compact
t0 = time.time()
Xe2, Xf2, Xs2 = descriptor_all2_R.compute_all_descriptors(
    pos, lat, dict(), 1.0, all_ls, radial_pairs, r_c, True, dummy_coef
)
t1 = time.time()
print(f"Time for descriptor_all2_R: {t1-t0:.4f} s")

# Verify
diff_e = np.max(np.abs(e_baseline.flatten() - Xe2.flatten()))
diff_f = np.max(np.abs(f_baseline.flatten() - Xf2.flatten()))

print(f"Energy Max Diff: {diff_e:.2e}")
print(f"Force Max Diff:  {diff_f:.2e}")

if diff_e < 1e-10 and diff_f < 1e-10:
    print("SUCCESS: Arrays match perfectly.")
else:
    print("ERROR: Mismatch found!")
