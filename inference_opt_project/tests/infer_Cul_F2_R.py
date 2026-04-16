"""
infer_Cul_F2_R.py
-----------------
Inference API using descriptor_all_R backend.

Loads models/linear_model_R.npz and predicts energy / forces / stress
for a single structure from a POSCAR, vasprun.xml, or raw arrays.
Reports elapsed time, number of atoms, and time per atom.

Usage:
    python infer_Cul_F2_R.py POSCAR_001
    python infer_Cul_F2_R.py path/to/vasprun.xml
    python infer_Cul_F2_R.py POSCAR_001 --model models/linear_model_R.npz --stress
    python infer_Cul_F2_R.py              # auto-finds first POSCAR_*

API usage:
    from infer_Cul_F2_R import load_model, predict_structure, predict_from_poscar

    model = load_model("models/linear_model_R.npz")

    # from Cartesian positions + row-vector lattice
    out = predict_structure(positions, lattice, model,
                            compute_force=True, compute_stress=False)
    print(out["energy"])            # float [eV]
    print(out["forces"])            # (N, 3) [eV/Ang]
    print(out["stress"])            # ndarray or None
    print(out["n_atoms"])           # int
    print(out["elapsed_sec"])       # float [s]
    print(out["time_per_atom_ms"])  # float [ms/atom]

    # from POSCAR
    out = predict_from_poscar("POSCAR_001", model)
"""

import argparse
import sys
import time
import glob as _glob

import numpy as np

import descriptor_all_R as descriptor_all

from cul_f2_r_utils import (
    load_model,
    load_eigenvectors,
    read_dataset_xml,
)

__all__ = [
    "load_model",
    "predict_structure",
    "predict_from_poscar",
    "predict_from_xml",
]


# =========================================================
# POSCAR reader
# =========================================================
def read_poscar(poscar_path):
    """
    Parse a VASP POSCAR/CONTCAR file.
    Returns (positions, lattice, elements).
      positions : (N, 3) Cartesian [Angstrom]
      lattice   : (3, 3) row vectors [Angstrom]
      elements  : list[str] length N
    """
    with open(poscar_path) as fh:
        lines = fh.readlines()

    scale = float(lines[1].strip())
    lattice = np.array(
        [[float(x) for x in lines[2 + i].split()] for i in range(3)],
        dtype=np.float64,
    ) * scale

    tok6 = lines[5].split()
    if tok6[0].lstrip("-").replace(".", "", 1).isdigit():
        species    = None
        counts     = [int(t) for t in tok6]
        coord_idx  = 6
    else:
        species    = tok6
        counts     = [int(t) for t in lines[6].split()]
        coord_idx  = 7

    n_atoms = sum(counts)

    coord_header = lines[coord_idx].strip()
    if coord_header[0].lower() == "s":
        coord_idx   += 1
        coord_header = lines[coord_idx].strip()

    direct = coord_header[0].lower() == "d"

    raw = np.array(
        [[float(x) for x in lines[coord_idx + 1 + i].split()[:3]]
         for i in range(n_atoms)],
        dtype=np.float64,
    )
    positions = (raw @ lattice) if direct else (raw * scale)

    if species is not None:
        elements = []
        for sym, cnt in zip(species, counts):
            elements.extend([sym] * cnt)
    else:
        elements = [f"X{i}" for i in range(n_atoms)]

    return positions, lattice, elements


# =========================================================
# Core inference (from raw arrays)
# =========================================================
def predict_structure(
    positions,
    lattice,
    model,
    data_dict,
    compute_force=True,
    compute_stress=False,
):
    """
    Predict energy, forces, and/or stress for a single structure.

    Parameters
    ----------
    positions : (N, 3) ndarray  Cartesian [Angstrom]
    lattice   : (3, 3) ndarray  row vectors [Angstrom]
    model     : dict from load_model()
    data_dict : dict  eigenvectors from load_eigenvectors()
    compute_force  : bool
    compute_stress : bool

    Returns
    -------
    dict:
      "energy"           : float [eV]
      "forces"           : (N, 3) [eV/Ang]  or None
      "stress"           : ndarray or None
      "n_atoms"          : int
      "elapsed_sec"      : float [s]
      "time_per_atom_ms" : float [ms/atom]
    """
    t0 = time.perf_counter()

    positions = np.asarray(positions, dtype=np.float64)
    lattice   = np.asarray(lattice,   dtype=np.float64)
    N = positions.shape[0]

    coef       = model["coef"]
    mean       = model["mean"]
    scale      = model["scale"]
    all_ls     = model["all_ls"]
    radial_pairs = model["RADIAL_PAIRS"]
    cutoff     = model["CUTOFF"]
    model_type = model["MODEL_TYPE"]
    max_p      = model["MAX_P"]
    poly_l_max = model["POLY_L_MAX"]
    energy_row_mode = model["ENERGY_ROW_MODE"]
    feature_order_mode = model["FEATURE_ORDER_MODE"]

    n_base_feat = len(all_ls)
    n_radial    = len(radial_pairs)

    # Compute descriptors using descriptor_all_R
    # (Same approach as in train_Cul_F2_R.py's compute_linear_data)
    Xe, Xf, Xs = descriptor_all.compute_all_descriptors(
        positions, lattice, data_dict, 1.0, all_ls, radial_pairs, cutoff,
        compute_force or compute_stress,
    )

    if not np.isfinite(Xe).all():
        raise ValueError("NaN/Inf in energy descriptors")

    # Feature order conversion
    from cul_f2_r_utils import (
        convert_feature_order,
        build_poly_feature_indices,
        build_feature_blocks,
        make_energy_row,
        generate_polynomial_features,
    )

    Xe = convert_feature_order(Xe, n_base_feat, n_radial, feature_order_mode)
    if compute_force or compute_stress:
        Xf = convert_feature_order(Xf, n_base_feat, n_radial, feature_order_mode)
        Xs = convert_feature_order(Xs, n_base_feat, n_radial, feature_order_mode)

    # Build feature blocks
    poly_feat_idx = build_poly_feature_indices(
        all_ls=all_ls, n_radial=n_radial,
        model_type=model_type, poly_l_max=poly_l_max,
    )

    d_struct = {"Xe_atom": Xe, "Xf_row": Xf, "Xs_row": Xs}
    xe_final, xf_final = build_feature_blocks(
        d_struct, mean, scale, poly_feat_idx, model_type, max_p
    )

    # Energy prediction
    x_e = make_energy_row(xe_final, mode=energy_row_mode).reshape(-1)
    energy = float(np.dot(x_e, coef))

    # Force prediction
    forces = None
    if compute_force:
        f_flat = xf_final @ coef
        forces = f_flat.reshape(N, 3)

    # Stress prediction
    stress = None
    if compute_stress:
        xs_shape = Xs.shape
        n_feat_raw = xs_shape[-1]
        Xs_flat = Xs.reshape(-1, n_feat_raw)

        if poly_feat_idx:
            Xe_sp = (np.clip(np.nan_to_num(Xe), -1e15, 1e15) - mean) / scale
            _, xs_poly = generate_polynomial_features(
                xe=Xe_sp, xf=Xs_flat / scale,
                poly_feat_idx=poly_feat_idx,
                max_p=max_p if model_type != 1 else 1,
                use_force=True,
            )
            xs_final = np.hstack([Xs_flat / scale] + xs_poly)
        else:
            xs_final = Xs_flat / scale

        s_flat = xs_final @ coef
        stress = s_flat.reshape(3, 3) if s_flat.size == 9 else s_flat

    elapsed = time.perf_counter() - t0

    return {
        "energy":           energy,
        "forces":           forces,
        "stress":           stress,
        "n_atoms":          N,
        "elapsed_sec":      elapsed,
        "time_per_atom_ms": elapsed / N * 1000.0,
    }


# =========================================================
# Convenience: infer from POSCAR
# =========================================================
def predict_from_poscar(poscar_path, model, data_dict, compute_force=True, compute_stress=False):
    """Load a POSCAR and run predict_structure."""
    positions, lattice, _ = read_poscar(poscar_path)
    return predict_structure(positions, lattice, model, data_dict,
                             compute_force=compute_force,
                             compute_stress=compute_stress)


# =========================================================
# Convenience: infer from vasprun.xml (with DFT reference)
# =========================================================
def predict_from_xml(xml_path, model, data_dict, compute_force=True, compute_stress=False):
    """
    Load a vasprun.xml, run inference, attach DFT reference values.

    Returns the same dict as predict_from_poscar, plus:
      "energy_dft" : float [eV]
      "forces_dft" : (N, 3) [eV/Ang]
      "stress_dft" : (3, 3) [kBar]
    """
    res = read_dataset_xml(xml_path)
    if res is None:
        raise ValueError(f"Cannot parse XML: {xml_path}")
    energy_dft, positions, lattice, forces_dft, stress_dft = res

    out = predict_structure(positions, lattice, model, data_dict,
                            compute_force=compute_force,
                            compute_stress=compute_stress)
    out["energy_dft"] = energy_dft
    out["forces_dft"] = forces_dft
    out["stress_dft"] = stress_dft
    return out


# =========================================================
# Pretty-print helper
# =========================================================
def _print_results(out, src=""):
    n_atoms = out["n_atoms"]
    forces  = out["forces"]

    print()
    print("=" * 52)
    print("  Inference Results  (descriptor_all_R backend)")
    if src:
        print(f"  {src}")
    print("=" * 52)
    print(f"  N atoms           : {n_atoms}")
    print(f"  Elapsed time      : {out['elapsed_sec'] * 1000:.3f} ms")
    print(f"  Time / atom       : {out['time_per_atom_ms']:.4f} ms/atom")
    print("-" * 52)
    print(f"  Energy (pred)     : {out['energy']:.8f} eV")
    print(f"  Energy / atom     : {out['energy'] / n_atoms:.8f} eV/atom")

    if out.get("energy_dft") is not None:
        e_err = out["energy"] - out["energy_dft"]
        print(f"  Energy (DFT)      : {out['energy_dft']:.8f} eV")
        print(f"  Energy error      : {e_err * 1000 / n_atoms:+.4f} meV/atom")

    if forces is not None:
        print(f"\n  Forces (pred) [eV/Ang], first 3 atoms:")
        for i, fv in enumerate(forces[:3]):
            print(f"    atom {i:>3d}: {fv[0]:+.6f}  {fv[1]:+.6f}  {fv[2]:+.6f}")

    if out.get("forces_dft") is not None and forces is not None:
        print(f"\n  Forces (DFT)  [eV/Ang], first 3 atoms:")
        for i, fv in enumerate(out["forces_dft"][:3]):
            print(f"    atom {i:>3d}: {fv[0]:+.6f}  {fv[1]:+.6f}  {fv[2]:+.6f}")
        rmse_f = float(np.sqrt(np.mean(
            (forces.reshape(-1) - out["forces_dft"].reshape(-1)) ** 2
        )))
        print(f"\n  Force RMSE        : {rmse_f:.6f} eV/Ang")

    if out.get("stress") is not None:
        s = np.asarray(out["stress"]).reshape(-1)
        labels = ["xx", "yy", "zz", "yz", "xz", "xy", "yx", "zx", "zy"][:s.size]
        print(f"\n  Stress (pred):")
        for lbl, val in zip(labels, s):
            print(f"    {lbl}: {val:+.6f}")

    print("=" * 52)


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run descriptor_all_R inference on a POSCAR or vasprun.xml."
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="POSCAR/CONTCAR or vasprun.xml. "
             "Omit to auto-search ./MLP_training_set/**/POSCAR_*.",
    )
    parser.add_argument(
        "--model", default="models/linear_model_R.npz",
        help="Path to .npz model (default: models/linear_model_R.npz)",
    )
    parser.add_argument("--no-force",  action="store_true", help="Hide force output.")
    parser.add_argument("--stress",    action="store_true", help="Show stress tensor.")
    args = parser.parse_args()

    compute_force  = not args.no_force
    compute_stress = args.stress

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)

    print(f"Loading eigenvectors from eigenvectors_R.npz ...")
    data_dict = load_eigenvectors("eigenvectors_R.npz")

    input_path = args.input
    if input_path is None:
        found = sorted(_glob.glob("./MLP_training_set/**/POSCAR*", recursive=True))
        if found:
            input_path = found[0]
            print(f"No input specified. Using: {input_path}")
        else:
            print("No input file specified and no POSCAR found. Exiting.")
            sys.exit(1)

    print(f"\nRunning inference on: {input_path}")

    if input_path.endswith(".xml"):
        out = predict_from_xml(input_path, model, data_dict,
                               compute_force=compute_force,
                               compute_stress=compute_stress)
    else:
        out = predict_from_poscar(input_path, model, data_dict,
                                  compute_force=compute_force,
                                  compute_stress=compute_stress)

    _print_results(out, src=input_path)
