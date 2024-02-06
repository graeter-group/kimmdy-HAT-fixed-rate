import numpy as np
from pathlib import Path
from tqdm import tqdm

version = 0.4  # truncated from HAT reaction plugin

def metas_to_ds(
    meta_files,
    max_dist,
    min_dist,
    opt,
    scale=False,
    old_scale=None,
    mask_energy=True,
    oneway=False,
):
    # Load energies
    meta_dicts1 = []
    meta_dicts2 = []
    energies1 = []
    energies2 = []
    pdbs1 = []

    for meta_f in tqdm(meta_files, "Loading data", mininterval=2):
        meta = np.load(meta_f, allow_pickle=True)
        meta_d = np.expand_dims(meta[meta.files[0]], axis=0)[0]

        m1 = meta_d.copy()
        m2 = meta_d.copy()
        m1["direction"] = 1
        m2["direction"] = 2
        meta_dicts1.append(m1)
        meta_dicts2.append(m2)
        name = meta_f.stem
        if "#" in name:  # duplicated meta files w/o dup pdbs
            name = name.rstrip("0123456789").rstrip("#")
        assert (
            not "#" in name
        ), f"ERROR duplicating pdb from {name}\nold name: {meta_f.stem}"

        pdbs1.append(
            (
                str(meta_f.parent / (name + "_1.pdb")),
                str(meta_f.parent / (name + "_2.pdb")),
            )
        )

        # filter translation
        if max_dist is not None:
            if meta_d["translation"] > max_dist:
                energies1.append(np.NaN)
                energies2.append(np.NaN)
                continue
        if min_dist is not None:
            if meta_d["translation"] < min_dist:
                energies1.append(np.NaN)
                energies2.append(np.NaN)
                continue

        # read energies
        if opt:
            if "e_s_opt" in meta_d and "e_ts_opt" in meta_d and "e_e_opt" in meta_d:
                energies1.append(meta_d["e_ts_opt"] - meta_d["e_s_opt"])
                energies2.append(meta_d["e_ts_opt"] - meta_d["e_e_opt"])
                continue
        else:
            if "e_max" in meta_d and "e_00" in meta_d and "e_10" in meta_d:
                energies1.append(meta_d["e_max"] - meta_d["e_00"])
                energies2.append(meta_d["e_max"] - meta_d["e_10"])
                continue

        energies1.append(np.NaN)
        energies2.append(np.NaN)

    if oneway:
        pdbs = np.array(pdbs1, dtype=str)
        energies = np.array(energies1, dtype=float)
        meta_dicts = np.array(meta_dicts1)
        metas_masked = np.array(meta_files)
    else:
        pdbs2 = [(j, i) for i, j in pdbs1]
        pdbs = np.array(pdbs1 + pdbs2, dtype=str)
        energies = np.array(energies1 + energies2, dtype=float)
        meta_dicts = np.array(meta_dicts1 + meta_dicts2)
        metas_masked = np.array(meta_files + meta_files)
    mask = np.logical_not(np.isnan(energies))

    if mask_energy:
        energies = energies[mask]
        pdbs = pdbs[mask]
        meta_dicts = meta_dicts[mask]
        metas_masked = metas_masked[mask]

    print(f"Loaded {len(energies)} systems!")

    mean = 0
    std = 1
    if old_scale is not None:
        mean = old_scale[0]
        std = old_scale[1]
        energies = (energies - mean) / std
        print(f"Old scale used: mean {mean}, std {std}")

    elif scale:
        mean = energies.mean()
        std = energies.std()
        energies = (energies - mean) / std
        print(f"Scaled, mean {mean}, std {std}")
    else:
        print("Not scaled")
    scale_t = (mean, std)

    return energies, pdbs, scale_t, meta_dicts, metas_masked

