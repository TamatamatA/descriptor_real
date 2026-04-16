"""
train_Cul_F2_R.py
-----------------
Train a linear (ridge) MLP model using descriptors from descriptor_all_R.

Usage:
    python train_Cul_F2_R.py

Output:
    models/linear_model_R.npz  -- saved model
"""

import glob
import numpy as np
import time

from cul_f2_r_utils import (
    load_eigenvectors,
    get_all_ls_from_npz,
    compute_linear_data,
    fit_scaler,
    build_design_matrix,
    fit_ridge_cholesky,
    evaluate_dataset,
    save_model,
    predict_energy_force,
)

# =========================================================
# Configuration
# =========================================================
CUTOFF = 7.0

MODEL_TYPE = 1
MAX_P = 1
POLY_L_MAX = 1

GTINV_MAXL = {
    1: 0,
    2: 11,
    3: 11,
}

INCLUDE_FORCE_TRAIN = False
INCLUDE_STRESS_TRAIN = False

EVAL_FORCE_RMSE = True
EVAL_STRESS_RMSE = False

DESCRIPTOR_COMPUTE_FORCE = INCLUDE_FORCE_TRAIN or EVAL_FORCE_RMSE

RIDGE_ALPHA = 1e-3
USE_STANDARDIZE = True

DATASET_TRAIN_GLOB = "./polymlp_datasets/train2/*.xml"
DATASET_TEST_GLOB  = "./polymlp_datasets/test2/*.xml"

RADIAL_PAIRS = [
    (3.0, 0.0),
    (3.0, 3.0),
    (3.0, 6.0),
    (0.0, 0.0),
]

ENERGY_ROW_MODE   = "sum"
FEATURE_ORDER_MODE = "as_is"

MODEL_SAVE_PATH   = "models/linear_model_R.npz"
EIGENVECTORS_PATH = "eigenvectors_R.npz"

SAVE_DESIGN_MATRIX = False
# =========================================================


def main():
    t_start = time.perf_counter()

    # ----------------------------------------------------------
    # Load eigenvectors and build all_ls
    # ----------------------------------------------------------
    data_dict = load_eigenvectors(EIGENVECTORS_PATH)
    all_ls = get_all_ls_from_npz(data_dict, GTINV_MAXL)

    # ----------------------------------------------------------
    # Compute descriptors for training set
    # ----------------------------------------------------------
    train_files = sorted(glob.glob(DATASET_TRAIN_GLOB, recursive=True))
    print(f"[-] Train Files: {len(train_files)}")
    train_linear_data = compute_linear_data(
        file_list=train_files,
        all_ls=all_ls,
        radial_pairs=RADIAL_PAIRS,
        cutoff=CUTOFF,
        feature_order_mode=FEATURE_ORDER_MODE,
        compute_force=DESCRIPTOR_COMPUTE_FORCE,
        data_dict=data_dict,
    )

    min_e_per_atom = min(d["E"] / d["N"] for d in train_linear_data)
    print(f"\nTrain min_e_per_atom for weight: {min_e_per_atom:.12f} eV/atom")

    # ----------------------------------------------------------
    # Standardization
    # ----------------------------------------------------------
    if USE_STANDARDIZE:
        mean, scale = fit_scaler(
            train_linear_data,
            energy_row_mode=ENERGY_ROW_MODE,
            include_force_train=INCLUDE_FORCE_TRAIN,
        )
    else:
        n_feat = len(all_ls) * len(RADIAL_PAIRS)
        mean  = np.zeros(n_feat, dtype=np.float64)
        scale = np.ones(n_feat,  dtype=np.float64)
        print(f"  USE_STANDARDIZE: False  (identity mean/scale, n_feat={n_feat})")

    # ----------------------------------------------------------
    # Build design matrix
    # ----------------------------------------------------------
    X_train, y_train, _ = build_design_matrix(
        linear_data=train_linear_data,
        all_ls=all_ls, radial_pairs=RADIAL_PAIRS,
        mean=mean, scale=scale,
        min_e_per_atom=min_e_per_atom,
        model_type=MODEL_TYPE, max_p=MAX_P, poly_l_max=POLY_L_MAX,
        energy_row_mode=ENERGY_ROW_MODE,
        feature_order_mode=FEATURE_ORDER_MODE,
        include_force_train=INCLUDE_FORCE_TRAIN,
    )

    print(f"  X_train.shape = {X_train.shape}")
    print(f"  y_train.shape = {y_train.shape}")

    if SAVE_DESIGN_MATRIX:
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    print("\nPhase 4: Ridge Regression Training (Cholesky)...")
    print(f"  RIDGE_ALPHA: {RIDGE_ALPHA}")
    coef = fit_ridge_cholesky(X_train, y_train, RIDGE_ALPHA)

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    evaluate_dataset(
        linear_data=train_linear_data,
        all_ls=all_ls, radial_pairs=RADIAL_PAIRS,
        mean=mean, scale=scale, coef=coef,
        model_type=MODEL_TYPE, max_p=MAX_P, poly_l_max=POLY_L_MAX,
        energy_row_mode=ENERGY_ROW_MODE,
        eval_force=EVAL_FORCE_RMSE, eval_stress=EVAL_STRESS_RMSE,
        label="TRAIN",
    )

    test_files = sorted(glob.glob(DATASET_TEST_GLOB, recursive=True))
    print(f"\n[-] Test Files: {len(test_files)}")
    if len(test_files) > 0:
        test_linear_data = compute_linear_data(
            file_list=test_files,
            all_ls=all_ls, radial_pairs=RADIAL_PAIRS,
            cutoff=CUTOFF,
            feature_order_mode=FEATURE_ORDER_MODE,
            compute_force=DESCRIPTOR_COMPUTE_FORCE,
            data_dict=data_dict,
        )
        evaluate_dataset(
            linear_data=test_linear_data,
            all_ls=all_ls, radial_pairs=RADIAL_PAIRS,
            mean=mean, scale=scale, coef=coef,
            model_type=MODEL_TYPE, max_p=MAX_P, poly_l_max=POLY_L_MAX,
            energy_row_mode=ENERGY_ROW_MODE,
            eval_force=EVAL_FORCE_RMSE, eval_stress=EVAL_STRESS_RMSE,
            label="TEST",
        )

    # ----------------------------------------------------------
    # Save model
    # ----------------------------------------------------------
    save_model(
        model_path=MODEL_SAVE_PATH,
        coef=coef, mean=mean, scale=scale,
        metadata={
            "all_ls": all_ls,
            "RADIAL_PAIRS": RADIAL_PAIRS,
            "CUTOFF": CUTOFF,
            "MODEL_TYPE": MODEL_TYPE,
            "MAX_P": MAX_P,
            "POLY_L_MAX": POLY_L_MAX,
            "GTINV_MAXL": GTINV_MAXL,
            "ENERGY_ROW_MODE": ENERGY_ROW_MODE,
            "FEATURE_ORDER_MODE": FEATURE_ORDER_MODE,
        },
    )

    # ----------------------------------------------------------
    # Consistency check: utils path vs infer path
    # ----------------------------------------------------------
    print("\n--- Consistency check (train structure #0) ---")
    _run_consistency_check(
        train_linear_data=train_linear_data,
        all_ls=all_ls, mean=mean, scale=scale, coef=coef,
        model_save_path=MODEL_SAVE_PATH,
        data_dict=data_dict,
    )

    print(f"\nDone. Total elapsed time: {time.perf_counter() - t_start:.1f} s")


def _run_consistency_check(
    train_linear_data, all_ls, mean, scale, coef, model_save_path, data_dict
):
    """
    Compare predict_energy_force (utils) with infer_Cul_F2_R.predict_structure
    on the first training structure.
    predict_energy_force returns e_pred as per-atom; multiply by N for cell energy.
    Force sign: Xf_row already encodes -dXe/dR, so F_pred = Xf_final @ coef.
    """
    try:
        from infer_Cul_F2_R import load_model, predict_structure
    except ImportError:
        print("  [skip] infer_Cul_F2_R.py not found; skipping consistency check.")
        return

    from cul_f2_r_utils import read_dataset_xml

    d0 = train_linear_data[0]
    N0 = d0["N"]

    ref = predict_energy_force(
        linear_data=[d0],
        all_ls=all_ls, radial_pairs=RADIAL_PAIRS,
        mean=mean, scale=scale, coef=coef,
        model_type=MODEL_TYPE, max_p=MAX_P, poly_l_max=POLY_L_MAX,
        energy_row_mode=ENERGY_ROW_MODE,
        eval_force=True,
    )
    e_pred_ref_cell = float(ref["e_pred"][0]) * N0
    f_pred_ref = ref["f_pred"].reshape(N0, 3) if "f_pred" in ref else None

    model = load_model(model_save_path)
    res = read_dataset_xml(d0["file"])
    if res is None:
        print(f"  [skip] Cannot re-read {d0['file']}")
        return
    _, pos, lat, _frc, _ = res

    out = predict_structure(pos, lat, model, data_dict, compute_force=True, compute_stress=False)

    e_ok = np.isclose(e_pred_ref_cell, out["energy"], rtol=1e-6, atol=1e-8)
    print(f"  Energy (utils):  {e_pred_ref_cell:.8f} eV")
    print(f"  Energy (infer):  {out['energy']:.8f} eV")
    print(f"  Energy match:    {e_ok}")

    if f_pred_ref is not None and out["forces"] is not None:
        f_ok = np.allclose(f_pred_ref, out["forces"], rtol=1e-5, atol=1e-7)
        f_max_diff = float(np.max(np.abs(f_pred_ref - out["forces"])))
        print(f"  Force max diff:  {f_max_diff:.3e} eV/Ang")
        print(f"  Force match:     {f_ok}")

    if not e_ok:
        print("  WARNING: energy mismatch between utils and infer paths!")


if __name__ == "__main__":
    main()
