"""
cul_f2_r_utils.py
-----------------
Shared utilities for train_Cul_F2_R.py and infer_Cul_F2_R.py.
All functions are ported from Cul_F2_R.py with minimal changes.

Force sign convention (verified from descriptor_all_R.cpp):

  descriptor_all_R returns Xf where:
    Xf[3*j, feat] = sum_i  -lambda_i * d(anlm_i)/dR_j
                  = -d(Xe_i)/dR_j   (already negative)

  Therefore:
    Xf_row[j] @ coef  =  -dXe/dR_j @ coef  =  -dE/dR_j  =  F_j  (physical force)

  So:  F_pred = Xf_final @ coef   (NO extra minus sign needed)

  The original Cul_F2_R.py applied two sign flips that cancelled:
    Xf = -Xf          (flip 1)
    "Xf_row": -Xf     (flip 2 -> back to original)
  and then used f_pred = xf_final @ coef, which was correct.
  We store Xf_row as-is (no flip) and likewise use F_pred = Xf_final @ coef.
"""

import xml.etree.ElementTree as ET
import glob
import numpy as np
import time
import itertools
import gc

import descriptor_all_R as descriptor_all


# =========================================================
# ridge solver (Cholesky / LAPACK posv)
# =========================================================
def solve_linear_equation_posv(A, b):
    from scipy.linalg.lapack import get_lapack_funcs
    (posv,) = get_lapack_funcs(("posv",), (A, b))
    if A.flags["C_CONTIGUOUS"]:
        _, x, info = posv(A.T, b, lower=False, overwrite_a=False, overwrite_b=False)
    else:
        _, x, info = posv(A, b, lower=False, overwrite_a=False, overwrite_b=False)
    if info != 0:
        raise RuntimeError(
            "Cholesky solve failed (X.T @ X + alpha I may be singular). "
            "Try increasing RIDGE_ALPHA."
        )
    return x


def fit_ridge_cholesky(X, y, alpha):
    xtx = X.T @ X
    xty = X.T @ y
    n_features = xtx.shape[0]
    xtx.flat[:: n_features + 1] += alpha
    coef = solve_linear_equation_posv(xtx, xty)
    return coef


# =========================================================
# I/O
# =========================================================
def read_dataset_xml(filename):
    """Read a VASP vasprun.xml and return (energy, positions, lattice, forces, stress)."""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        node_calc = root.find("calculation")
        node_energy = node_calc.find("energy") if node_calc is not None else None
        if node_energy is None or len(list(node_energy)) < 2:
            return None
        energy = float(list(node_energy)[1].text)

        lattice = []
        for v in root.find(".//varray[@name='basis']").findall("v"):
            lattice.append([float(x) for x in v.text.split()])
        lattice = np.array(lattice, dtype=np.float64)

        pos_frac = []
        for v in root.find(".//varray[@name='positions']").findall("v"):
            pos_frac.append([float(x) for x in v.text.split()])
        positions = np.dot(np.array(pos_frac, dtype=np.float64), lattice)

        forces = []
        for v in root.find(".//varray[@name='forces']").findall("v"):
            forces.append([float(x) for x in v.text.split()])
        forces = np.array(forces, dtype=np.float64)

        stress = []
        node_s = root.find(".//varray[@name='stress']")
        if node_s is not None:
            for v in node_s.findall("v"):
                stress.append([float(x) for x in v.text.split()])
        else:
            stress = np.zeros((3, 3), dtype=np.float64)
        stress = np.array(stress, dtype=np.float64)

        return energy, positions, lattice, forces, stress
    except Exception:
        return None


# =========================================================
# ls pattern utilities
# =========================================================
def generate_all_ls_patterns(l_max_settings):
    all_patterns = []
    for order in sorted(l_max_settings.keys()):
        current_l_max = l_max_settings[order]
        if order == 1:
            all_patterns.append([0])
            continue
        l_range = range(0, current_l_max + 1)
        for ls in itertools.combinations_with_replacement(l_range, r=order):
            if sum(ls) % 2 == 0:
                all_patterns.append(list(ls))
    return all_patterns


def find_valid_ls(data_dict, l_max_settings):
    raw_ls_patterns = generate_all_ls_patterns(l_max_settings)

    valid_ls = []
    skipped_count = 0

    actual_keys = list(data_dict.keys())
    first_key = actual_keys[0] if actual_keys else ""
    use_bracket = "[" in first_key

    for ls in raw_ls_patterns:
        if use_bracket:
            key = f"l_list[{', '.join(map(str, ls))}]"
        else:
            s = str(tuple(ls))
            if len(ls) == 1:
                s = s.replace(",", ", ")
            key = f"l_list{s}"

        if key in data_dict:
            valid_ls.append(ls)
        else:
            if skipped_count < 3:
                print(f"  [Debug] Searching for: '{key}' ... Not Found")
            skipped_count += 1

    print("-" * 30)
    print("[Summary of Descriptors]")
    print(f"Total patterns generated : {len(raw_ls_patterns)}")
    print(f"Valid patterns (in NPZ)  : {len(valid_ls)}")
    print(f"Skipped patterns         : {skipped_count}")
    print("-" * 30)

    return valid_ls


def get_all_ls_from_npz(data_dict, gtinv_maxl):
    """Build all_ls from eigenvectors.npz keys using given GTINV_MAXL dict."""
    return find_valid_ls(data_dict, gtinv_maxl)


# =========================================================
# feature order conversion
# =========================================================
def radial_major_to_feature_major(x, n_base_feat, n_radial):
    orig_shape = x.shape
    x2 = x.reshape(*orig_shape[:-1], n_radial, n_base_feat)
    x2 = np.swapaxes(x2, -2, -1)
    return x2.reshape(*orig_shape[:-1], n_base_feat * n_radial)


def convert_feature_order(x, n_base_feat, n_radial, mode):
    if mode == "as_is":
        return x
    elif mode == "radial_to_feature":
        return radial_major_to_feature_major(x, n_base_feat, n_radial)
    else:
        raise ValueError(f"Unknown FEATURE_ORDER_MODE: {mode}")


# =========================================================
# polynomial features
# =========================================================
def build_poly_feature_indices(all_ls, n_radial, model_type, poly_l_max):
    if model_type == 1:
        return []

    if model_type == 2:
        target_ls_idx = list(range(len(all_ls)))
    elif model_type == 4:
        target_ls_idx = [
            i for i, ls in enumerate(all_ls)
            if all(l <= poly_l_max for l in ls)
        ]
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

    poly_feat_idx = []
    for i in target_ls_idx:
        poly_feat_idx.extend(range(i * n_radial, (i + 1) * n_radial))
    return poly_feat_idx


def generate_polynomial_features(xe, xf, poly_feat_idx, max_p, use_force):
    xe_poly_list = []
    xf_poly_list = []

    if max_p < 2 or len(poly_feat_idx) == 0:
        return xe_poly_list, xf_poly_list

    for order in range(2, max_p + 1):
        print(f"    Generating polynomial order {order} ...")
        for idx_tuple in itertools.combinations_with_replacement(poly_feat_idx, order):
            cols_e = [xe[:, idx:idx + 1] for idx in idx_tuple]

            e_prod = cols_e[0].copy()
            for col in cols_e[1:]:
                e_prod *= col
            xe_poly_list.append(e_prod)

            if use_force:
                cols_f = [xf[:, idx:idx + 1] for idx in idx_tuple]
                cols_e_ext = [np.repeat(col, 3, axis=0) for col in cols_e]

                f_prod = np.zeros_like(cols_f[0])
                for t in range(order):
                    term = cols_f[t].copy()
                    for s in range(order):
                        if s != t:
                            term *= cols_e_ext[s]
                    f_prod += term
                xf_poly_list.append(f_prod)

    return xe_poly_list, xf_poly_list


# =========================================================
# weighting
# =========================================================
def set_weight_energy_data(energy, total_n_atoms, min_e_per_atom=None):
    e_per_atom = energy / total_n_atoms
    if min_e_per_atom is None:
        min_e_per_atom = e_per_atom

    e_th1 = min_e_per_atom * 0.75
    e_th2 = min_e_per_atom * 0.50

    weight_e = 1.0
    if e_per_atom > e_th1:
        weight_e = 0.5
    if e_per_atom > e_th2:
        weight_e = 0.3
    if e_per_atom > 0.0:
        weight_e = 0.1

    return weight_e


def set_weight_force_data(forces, tol=1e-12):
    weight_f = np.abs(forces).copy()
    weight_f[weight_f < tol] = tol
    weight_f = 1.0 / weight_f
    weight_f[weight_f > 1.0] = 1.0
    return weight_f


# =========================================================
# descriptor helper
# =========================================================
def load_eigenvectors(path="eigenvectors_R.npz"):
    print("Loading eigenvector data and normalizing memory layout...")
    try:
        raw_data = np.load(path)
        data_dict = {}
        for k, v in raw_data.items():
            data_dict[k] = np.ascontiguousarray(v.real, dtype=np.float64).flatten()
        return data_dict
    except Exception:
        print(f"Error: {path} not found.")
        raise SystemExit(1)


def compute_linear_data(file_list, all_ls, radial_pairs, cutoff,
                        feature_order_mode, compute_force=True, data_dict=None):
    """
    Compute descriptors for a list of XML files.

    Parameters
    ----------
    file_list : list
        List of vasprun.xml files
    all_ls : list
        Descriptor angular momentum channels
    radial_pairs : list
        Radial basis parameters
    cutoff : float
        Cutoff radius
    feature_order_mode : str
        Feature ordering mode
    compute_force : bool
        Whether to compute force descriptors
    data_dict : dict or None
        Eigenvectors dictionary from load_eigenvectors().

    Xf_row is stored exactly as descriptor_all_R returns it:
      Xf_row[3*j, feat] = -d(Xe)/dR_j   (already carries the minus sign)
    Force prediction: F_pred = Xf_final @ coef   (NO extra minus needed)
    """
    if data_dict is None:
        data_dict = {}

    linear_data = []

    print("\nPhase 1: Computing Descriptors (E, F, S)...")
    t0 = time.time()
    descriptor_time_total = 0.0

    n_base_feat = len(all_ls)
    n_radial = len(radial_pairs)

    for i, f in enumerate(file_list):
        res = read_dataset_xml(f)
        if res is None:
            continue
        energy, pos, lat, frc, strss = res

        try:
            t_desc0 = time.time()

            Xe, Xf, Xs = descriptor_all.compute_all_descriptors(
                pos, lat, data_dict, 1.0, all_ls, radial_pairs, cutoff, compute_force,
            )
            descriptor_time_total += (time.time() - t_desc0)

            if (not np.isfinite(Xe).all()
                    or not np.isfinite(Xf).all()
                    or not np.isfinite(Xs).all()):
                raise ValueError(f"NaN detected in {f}")

            Xe = convert_feature_order(Xe, n_base_feat, n_radial, feature_order_mode)
            Xf = convert_feature_order(Xf, n_base_feat, n_radial, feature_order_mode)
            Xs = convert_feature_order(Xs, n_base_feat, n_radial, feature_order_mode)

            # Xf from descriptor_all_R already encodes -d(Xe)/dR.
            # Force prediction: F_pred = Xf_final @ coef  (no extra minus)
            linear_data.append({
                "Xe_atom": Xe,    # shape: (N, n_feat)
                "Xf_row": Xf,     # shape: (3N, n_feat), encodes -d(Xe)/dR
                "Xs_row": Xs,
                "E": energy,
                "F": frc,         # shape: (N, 3), DFT reference
                "S": strss,
                "N": len(pos),
                "V": np.linalg.det(lat),
                "file": f,
            })

        except Exception as e:
            print(f"  Skipping {f}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1} files... "
                f"(elapsed={time.time() - t0:.1f}s, "
                f"descriptor_only={descriptor_time_total:.1f}s)"
            )

    if not linear_data:
        print("Error: No valid data computed.")
        raise SystemExit(1)

    print(f"  Descriptor-only total time: {descriptor_time_total:.3f} s")
    print(f"  Avg descriptor time / structure: "
          f"{descriptor_time_total / len(linear_data):.6f} s")

    return linear_data


# =========================================================
# row construction
# =========================================================
def make_energy_row(xe_atom, mode="sum"):
    if mode == "sum":
        return np.sum(xe_atom, axis=0, keepdims=True)
    elif mode == "mean":
        return np.mean(xe_atom, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown ENERGY_ROW_MODE: {mode}")


def fit_scaler(linear_data, energy_row_mode, include_force_train):
    print("\nPhase 2: Standardization...")

    energy_rows = []
    for d in linear_data:
        xe_atom = np.clip(np.nan_to_num(d["Xe_atom"]), -1e15, 1e15)
        x_e = make_energy_row(xe_atom, mode=energy_row_mode).reshape(-1)
        energy_rows.append(x_e)

    energy_rows = np.vstack(energy_rows)

    scale = np.std(energy_rows, axis=0)
    scale_threshold = 1e-10 if include_force_train else 1e-20
    scale[np.abs(scale) < scale_threshold] = 1.0
    mean = np.zeros_like(scale)

    print(f"  Mean max: {np.max(np.abs(mean)):.2e}, "
          f"Scale max: {np.max(np.abs(scale)):.2e}")

    del energy_rows
    gc.collect()
    return mean, scale


def build_feature_blocks(d, mean, scale, poly_feat_idx, model_type, max_p):
    """
    Standardize and optionally expand features for one structure.
    Returns (xe_final, xf_final) where xf_final encodes -d(Xe)/dR / scale.
    """
    xe_atom_raw = np.clip(np.nan_to_num(d["Xe_atom"]), -1e15, 1e15)
    xf_row_raw  = np.clip(np.nan_to_num(d["Xf_row"]),  -1e15, 1e15)

    xe_atom = (xe_atom_raw - mean) / scale
    xf_row  = xf_row_raw / scale   # mean NOT subtracted for gradient rows

    xe_poly_list, xf_poly_list = generate_polynomial_features(
        xe=xe_atom,
        xf=xf_row,
        poly_feat_idx=poly_feat_idx,
        max_p=max_p if model_type != 1 else 1,
        use_force=True,
    )

    xe_final = np.hstack([xe_atom] + xe_poly_list) if xe_poly_list else xe_atom
    xf_final = np.hstack([xf_row]  + xf_poly_list) if xf_poly_list else xf_row

    return xe_final, xf_final


def build_design_matrix(
    linear_data, all_ls, radial_pairs, mean, scale, min_e_per_atom,
    model_type, max_p, poly_l_max, energy_row_mode, feature_order_mode,
    include_force_train,
):
    print("\nPhase 3: Stacking Matrix...")

    X_list, y_list = [], []
    e_indices, f_indices = [], []
    current_row = 0

    n_radial = len(radial_pairs)
    poly_feat_idx = build_poly_feature_indices(
        all_ls=all_ls, n_radial=n_radial,
        model_type=model_type, poly_l_max=poly_l_max,
    )

    print(f"  Base linear features: {len(all_ls) * n_radial}")
    print(f"  Selected for poly: {len(poly_feat_idx)} features")
    print(f"  ENERGY_ROW_MODE: {energy_row_mode}")
    print(f"  FEATURE_ORDER_MODE: {feature_order_mode}")
    print(f"  min_e_per_atom for weight: {min_e_per_atom:.12f} eV/atom")

    for idata, d in enumerate(linear_data):
        xe_final, xf_final = build_feature_blocks(
            d, mean, scale, poly_feat_idx, model_type, max_p
        )

        x_e_scaled = make_energy_row(xe_final, mode=energy_row_mode).reshape(-1)
        y_e = np.array([d["E"]], dtype=np.float64)

        w_e = set_weight_energy_data(
            energy=d["E"], total_n_atoms=d["N"],
            min_e_per_atom=min_e_per_atom,
        )

        X_list.append((x_e_scaled * w_e).reshape(1, -1))
        y_list.append(y_e * w_e)
        e_indices.append(current_row)
        current_row += 1

        if include_force_train:
            # Xf_row already encodes -d(Xe)/dR, so:
            #   xf_final @ coef  ≈  F_DFT
            # Training row is simply xf_final (no sign flip needed).
            y_f = d["F"].reshape(-1)
            w_f = set_weight_force_data(y_f)

            x_f = xf_final * w_f[:, None]
            y_f_w = y_f * w_f

            X_list.append(x_f)
            y_list.append(y_f_w)
            f_indices.extend(range(current_row, current_row + x_f.shape[0]))
            current_row += x_f.shape[0]

        if (idata + 1) % 50 == 0:
            print(f"  Stacked {idata + 1} structures")

    X_mat = np.vstack(X_list)
    y_vec = np.concatenate(y_list)

    meta = {"e_indices": e_indices, "f_indices": f_indices}
    return X_mat, y_vec, meta


# =========================================================
# prediction / evaluation
# =========================================================
def predict_energy_force(
    linear_data, all_ls, radial_pairs, mean, scale, coef,
    model_type, max_p, poly_l_max, energy_row_mode,
    eval_force=True, eval_stress=False,
):
    """
    Predict energy and force for a list of pre-computed linear_data entries.

    E_pred = (Xe_final.sum(axis=0)) @ coef
    F_pred = Xf_final @ coef
             Xf_row already encodes -d(Xe)/dR (see module docstring),
             so no extra minus sign is needed here.
    """
    n_radial = len(radial_pairs)
    poly_feat_idx = build_poly_feature_indices(
        all_ls=all_ls, n_radial=n_radial,
        model_type=model_type, poly_l_max=poly_l_max,
    )

    e_true_list, e_pred_list = [], []
    f_true_list, f_pred_list = [], []

    for d in linear_data:
        xe_final, xf_final = build_feature_blocks(
            d, mean, scale, poly_feat_idx, model_type, max_p
        )

        x_e = make_energy_row(xe_final, mode=energy_row_mode).reshape(-1)
        e_pred_cell = np.dot(x_e, coef)

        e_true_list.append(d["E"] / d["N"])
        e_pred_list.append(e_pred_cell / d["N"])

        if eval_force:
            # Xf_row already encodes -d(Xe)/dR, so F_pred = Xf_final @ coef
            f_pred = xf_final @ coef
            f_true = d["F"].reshape(-1)
            f_true_list.append(f_true)
            f_pred_list.append(f_pred)

    out = {
        "e_true": np.array(e_true_list),
        "e_pred": np.array(e_pred_list),
    }

    if eval_force:
        out["f_true"] = np.concatenate(f_true_list)
        out["f_pred"] = np.concatenate(f_pred_list)

    return out


def evaluate_dataset(
    linear_data, all_ls, radial_pairs, mean, scale, coef,
    model_type, max_p, poly_l_max, energy_row_mode,
    eval_force=True, eval_stress=False, label="TRAIN",
):
    from sklearn.metrics import mean_squared_error
    print(f"\nPhase 5: Evaluation on {label} set...")

    pred = predict_energy_force(
        linear_data=linear_data,
        all_ls=all_ls, radial_pairs=radial_pairs,
        mean=mean, scale=scale, coef=coef,
        model_type=model_type, max_p=max_p, poly_l_max=poly_l_max,
        energy_row_mode=energy_row_mode,
        eval_force=eval_force, eval_stress=eval_stress,
    )

    rmse_e = np.sqrt(mean_squared_error(pred["e_true"], pred["e_pred"]))

    print("=" * 55)
    print(f"{('PHYSICAL ERROR ANALYSIS: ' + label):^55}")
    print("=" * 55)
    print(f"  Energy RMSE : {rmse_e * 1000:16.8f} [meV/atom]")

    if eval_force and "f_true" in pred:
        rmse_f = np.sqrt(mean_squared_error(pred["f_true"], pred["f_pred"]))
        print(f"  Force  RMSE : {rmse_f:16.8f} [eV/Angstrom]")

    print("=" * 55)


# =========================================================
# model save / load
# =========================================================
def save_model(model_path, coef, mean, scale, metadata):
    """
    Save a trained linear model to a .npz file.

    metadata keys:
        all_ls, RADIAL_PAIRS, CUTOFF, MODEL_TYPE, MAX_P,
        POLY_L_MAX, GTINV_MAXL, ENERGY_ROW_MODE, FEATURE_ORDER_MODE
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

    all_ls = metadata["all_ls"]
    all_ls_flat    = np.array([v for ls in all_ls for v in ls], dtype=np.int32)
    all_ls_lengths = np.array([len(ls) for ls in all_ls],       dtype=np.int32)

    radial_pairs = np.array(metadata["RADIAL_PAIRS"], dtype=np.float64)

    gtinv_maxl = metadata["GTINV_MAXL"]
    gtinv_keys = np.array(list(gtinv_maxl.keys()),   dtype=np.int32)
    gtinv_vals = np.array(list(gtinv_maxl.values()), dtype=np.int32)

    np.savez(
        model_path,
        coef=coef, mean=mean, scale=scale,
        all_ls_flat=all_ls_flat, all_ls_lengths=all_ls_lengths,
        RADIAL_PAIRS=radial_pairs,
        CUTOFF=np.float64(metadata["CUTOFF"]),
        MODEL_TYPE=np.int32(metadata["MODEL_TYPE"]),
        MAX_P=np.int32(metadata["MAX_P"]),
        POLY_L_MAX=np.int32(metadata["POLY_L_MAX"]),
        GTINV_MAXL_KEYS=gtinv_keys, GTINV_MAXL_VALS=gtinv_vals,
        ENERGY_ROW_MODE=np.array(metadata["ENERGY_ROW_MODE"]),
        FEATURE_ORDER_MODE=np.array(metadata["FEATURE_ORDER_MODE"]),
    )
    print(f"[save_model] Saved to {model_path}")


def load_model(model_path):
    """
    Load a saved model and return a dict with all parameters ready to use.

    Returns dict with keys:
        coef, mean, scale, all_ls, RADIAL_PAIRS, CUTOFF,
        MODEL_TYPE, MAX_P, POLY_L_MAX, GTINV_MAXL,
        ENERGY_ROW_MODE, FEATURE_ORDER_MODE
    """
    data = np.load(model_path, allow_pickle=False)

    all_ls_flat    = data["all_ls_flat"]
    all_ls_lengths = data["all_ls_lengths"]
    all_ls = []
    offset = 0
    for length in all_ls_lengths:
        all_ls.append(list(map(int, all_ls_flat[offset: offset + length])))
        offset += length

    radial_pairs_arr = data["RADIAL_PAIRS"]
    radial_pairs = [(float(r[0]), float(r[1])) for r in radial_pairs_arr]

    gtinv_maxl = {int(k): int(v)
                  for k, v in zip(data["GTINV_MAXL_KEYS"], data["GTINV_MAXL_VALS"])}

    model = {
        "coef":               data["coef"],
        "mean":               data["mean"],
        "scale":              data["scale"],
        "all_ls":             all_ls,
        "RADIAL_PAIRS":       radial_pairs,
        "CUTOFF":             float(data["CUTOFF"]),
        "MODEL_TYPE":         int(data["MODEL_TYPE"]),
        "MAX_P":              int(data["MAX_P"]),
        "POLY_L_MAX":         int(data["POLY_L_MAX"]),
        "GTINV_MAXL":         gtinv_maxl,
        "ENERGY_ROW_MODE":    str(data["ENERGY_ROW_MODE"]),
        "FEATURE_ORDER_MODE": str(data["FEATURE_ORDER_MODE"]),
    }
    print(f"[load_model] Loaded from {model_path}")
    return model
