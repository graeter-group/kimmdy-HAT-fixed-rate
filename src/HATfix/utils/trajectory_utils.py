from itertools import combinations
import logging
import MDAnalysis.coordinates.timestep

import MDAnalysis as mda
import numpy as np
from scipy.spatial.transform import Rotation

from typing import Optional
import numpy.typing as npt

version = 0.6   # truncated from HAT reaction plugin


def check_cylinderclash(a, b, t, r_min=0.8, d_min=None, verbose=False):
    """Checks for atoms in a cylinder around a possible HAT reaction path.
    Cylinder begins/ends 10% after/before the end points.
    Ref.: https://geomalgorithms.com/a02-_lines.html

    Parameters
    ----------
    a : Union[np.ndarray, list]
        Center of cylinder base
    b : Union[np.ndarray, list]
        Center of cylinder top
    t : Union[np.ndarray, list]
        Testpoint, or list of testpoints
    d_min
    r_min : int, optional
        Radius of cylinder, by default 0.8

    Returns
    -------
    bool
        True if no points are inside the cylinder, False otherwise.
    """

    def _check_point(a, b, t, r_min, verbose):
        v = b - a  # apex to base
        w = t - a  # apex to testpoint
        c1 = np.dot(v, w)
        c2 = np.dot(v, v)
        x = c1 / c2  # percentage along axis

        tx = a + (x * v)  # projection of t on x
        r = np.linalg.norm(t - tx)

        if x < 0.1 or x > 0.9 or r > r_min:
            # point is outside cylinder
            return True
        if verbose:
            print(f"a = {a}, b = {b}, t = {t}")
            print(f"x = {x}, r = {r}")
        return False

    if isinstance(t, np.ndarray) and t.shape == (3,):
        return _check_point(a, b, t, r_min, verbose)

    elif (
        (isinstance(t, np.ndarray) and t.shape[1] == 3)
        or isinstance(t, list)
        and len(t[0]) == 3
    ):
        return all(
            [
                _check_point(np.array(a), np.array(b), np.array(p), r_min, verbose)
                for p in t
            ]
        )

    else:
        raise ValueError(f"Type/Shape of t wrong: {t}")

def find_radical_pos(
    center: mda.core.groups.Atom,
    bonded: mda.core.groups.AtomGroup,
):
    """Calculates possible radical positions of a given radical atom

    Parameters
    ----------
    center : mda.core.groups.Atom
        Radical atom
    bonded : mda.core.groups.AtomGroup
        Atom group of bonded atoms. From its length the geometry is predicted.

    Returns
    -------
    list
        List of radical positions, three dimensional arrays
    """
    scale_C = 1.10
    scale_N = 1.04
    scale_O = 0.97
    scale_S = 1.41

    if len(bonded) in [2, 3]:
        assert center.element in [
            "C",
            "N",
        ], f"Element {center.element} does not match bond number"

        if center.element == "N":
            scale = scale_N
        elif center.element == "C":
            scale = scale_C

        b_normed = []
        for b in bonded:
            b_vec = b.position - center.position
            b_vec_norm = b_vec / np.linalg.norm(b_vec)
            b_normed.append(b_vec_norm)

        midpoint = sum(b_normed)

        if len(bonded) == 3 and np.linalg.norm(midpoint) < 0.6:
            # flat structure -> two end positions:
            # midpoint: 109.5 -> ~1, 120 -> 0
            ab = bonded[1].position - bonded[0].position
            ac = bonded[2].position - bonded[0].position
            normal = np.cross(ab, ac)
            normal = normal / np.linalg.norm(normal)

            rad1 = center.position + (normal * scale)
            rad2 = center.position + (normal * (-1) * scale)
            rads = [rad1, rad2]

        else:
            # two bonds, or three in tetraeder:
            # -> mirror mean bond
            v = midpoint / np.linalg.norm(midpoint)
            rads = [center.position + (-1 * v * scale)]

        return rads

    # Radicals w/ only one bond:
    elif len(bonded) == 1:
        # suggest positions in a 109.5 degree cone
        assert center.element in ["O", "S"], "Element type does not match bond number"
        if center.element == "O":
            scale = scale_O
        elif center.element == "S":
            scale = scale_S

        b = bonded[0]
        b_vec = b.position - center.position
        b_vec = b_vec / np.linalg.norm(b_vec)
        rnd_vec = [1, 1, 1]  # to find a vector perpendicular to b_vec

        rnd_rot_ax = np.cross(b_vec, rnd_vec)
        rnd_rot_ax = rnd_rot_ax / np.linalg.norm(rnd_rot_ax)

        r1 = Rotation.from_rotvec(1.911 * rnd_rot_ax)  # 109.5 degree (as in EtOH)
        r2 = Rotation.from_rotvec(0.785 * b_vec)  # 45 degree

        ends = [r1.apply(b_vec)]  # up to 109.5

        for i in range(8):
            ends.append(r2.apply(ends[-1]))  # turn in 45d steps

        # norm and vec --> position
        ends = [(e / np.linalg.norm(e)) * scale + center.position for e in ends]

        return ends

    else:
        raise ValueError(f"Weired count of bonds: {list(bonded)}\n\tCorrect radicals?")


def extract_single_rad(
    u: mda.Universe,
    ts: MDAnalysis.coordinates.timestep,
    rad: mda.AtomGroup,
    bonded_rad: mda.AtomGroup,
    h_cutoff: float = 3,
    env_cutoff: float = 10,
) -> npt.NDArray:
    """Produces one cutout for each possible reaction around one given radical.

    Parameters
    ----------
    u
        Universe around the radical
    ts
        current timestep
    rad
        radical
    bonded_rad
        all atoms bound to the radical
    h_cutoff
        maximum distance a hydrogen can travel, by default 3
    env_cutoff
        size of cutout to make, by default 10

    Returns
    -------
    np.ndarray[dict]
        Array with one dict per reaction. Each dict hold start and end Universe,
        as well as meta data
    """
    env = u.atoms.select_atoms(
        f"point { str(rad.positions).strip('[ ]') } {env_cutoff}"
    )
    end_poss = find_radical_pos(rad[0], bonded_rad)
    hs = []
    for end_pos in end_poss:
        hs.append(
            env.select_atoms(
                f"point { str(end_pos).strip('[ ]') } {h_cutoff} and element H"
            )
        )
    hs = sum(hs) - bonded_rad  # exclude alpha-H

    clashes = np.empty((len(hs), len(end_poss)), dtype=bool)
    for h_idx, h in enumerate(hs):
        for end_idx, end_pos in enumerate(end_poss):
            clashes[h_idx, end_idx] = check_cylinderclash(
                end_pos, h.position, env.positions, r_min=0.8
            )

    cut_systems = np.zeros((len(hs),), dtype=object)
    min_translations = np.ones((len(hs),)) * 99

    # iterate over defined HAT reactions
    for h_idx, end_idx in zip(*np.nonzero(clashes)):
        end_pos = end_poss[end_idx]
        h = env.select_atoms(f"id {hs[h_idx].id}")

        translation = np.linalg.norm(end_pos - h.positions)
        # only keep reaction w/ smallest translation
        # there can be multiple end positions for one rad!
        if translation > min_translations[h_idx]:
            continue
        min_translations[h_idx] = translation

        other_atms = env - h - rad

        cut_systems[h_idx] = {
            "start_u": mda.core.universe.Merge(h, rad, other_atms),
            "end_u": mda.core.universe.Merge(h, rad, other_atms),
            "meta": {
                "translation": translation,
                "u1_name": rad[0].resname.lower() + "-sim",
                "u2_name": h[0].resname.lower() + "-sim",
                "trajectory": u._trajectory.filename,
                "frame": ts.frame,
                "indices": (*h.ids, *rad.ids, *other_atms.ids),
                "intramol": rad[0].residue == h[0].residue,
            },
        }

        # change H position in end universe
        cut_systems[h_idx]["end_u"].atoms[0].position = end_pos

        # hashes based on systems rather than subgroups, subgroubs would collide
        cut_systems[h_idx]["meta"]["hash_u1"] = abs(hash(cut_systems[h_idx]["start_u"]))
        cut_systems[h_idx]["meta"]["hash_u2"] = abs(hash(cut_systems[h_idx]["end_u"]))

    return cut_systems[np.nonzero(cut_systems)[0]]


def extract_subsystems(
    u: mda.Universe,
    rad_ids: list[str],
    h_cutoff: float = 3,
    env_cutoff: float = 7,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    rad_min_dist: float = 3,
    unique=False,
    cap: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),
) -> list:
    """Builds subsystems out of a trajectory for evaluation of HAT reaction
    either by DFT or a ML model.
    Aminoacids are optionally capped at peptide
    bonds resulting in amines and amides.
    Subsystems contain the reactive hydrogen at index 0 followed by the
    radical atom.
    Note: This adaptes the residue names in given universe to the break

    Parameters
    ----------
    u
        Main universe
    rad_idxs
        Indices of the two radical atoms
    h_cutoff
        Cutoff radius for hydrogen search around radical, by default 3
    env_cutoff
        Cutoff radius for local env, by default 7
    start
        For slicing the trajectory, by default None
    stop
        For slicing the trajectory, by default None
    step
        For slicing the trajectory, by default None
    unique
        If true, only keep one of every set of atoms.
    cap
        Whether or not the subsystems should be capped. If false, subsystems are
        created by cutting out a sphere with radius env_cutoff. Optional, default: True
    logger
        logger instance, optional

    Returns
    -------
    list
        List of capped subsystems
    """

    assert len(rad_ids) > 0, "Error: At least one radical must be given!"

    rads: list[mda.AtomGroup] = [u.select_atoms(f"id {rad}") for rad in rad_ids]

    # Delete bonds between radicals
    if len(rad_ids) > 1:
        combs = combinations(rad_ids, 2)
        for c in combs:
            try:
                u.delete_bonds([c])
            except ValueError:
                continue

    bonded_all = [u.select_atoms(f"bonded id {rad}") for rad in rad_ids]
    # remove rads
    bonded_all: list[mda.AtomGroup] = [b - sum(rads) for b in bonded_all]

    # correct residue of radicals to avoid residues w/ only 2 atoms
    # Necessary in case of backbone break other than peptide bond
    for rad, bonded_rad in zip(rads, bonded_all):
        if len(bonded_rad.residues) == 1:
            continue

        res_rad_org = rad[0].residue
        for bonded in bonded_rad:
            if bonded.residue == res_rad_org:
                # bonded to nothing else than the radical:
                if (bonded.bonded_atoms - rad).n_atoms == 0:
                    goal_res = bonded_rad.residues - rad[0].residue
                    assert len(goal_res) == 1
                    rad[0].residue = goal_res[0]
                    bonded_rad.residues = goal_res[0]

    cut_systems = {}

    for ts in u.trajectory[slice(start, stop, step)]:
        for i, (rad, bonded_rad) in enumerate(zip(rads, bonded_all)):
            # skip small distances
            skip = False
            for j, other_rad in enumerate(rads):
                if i == j:
                    continue
                if (
                    np.linalg.norm(rad.positions[0] - other_rad.positions[0])
                    < rad_min_dist
                ):
                    logger.debug(
                        f"Radical {rad} distance too small to {other_rad} in frame {ts.frame}, skipping.."
                    )
                    skip = True
            if skip:
                continue


            cut_frame = extract_single_rad(
                u, ts, rad, bonded_rad, h_cutoff, env_cutoff
            )

            for i, cut_sys_dict in enumerate(cut_frame):
                if unique:
                    new_i_hash = hash(cut_sys_dict["meta"]["indices"])
                else:
                    new_i_hash = str(i) + str(rad.indices) + str(ts.frame)

                # skip existing systems w/ bigger translation
                if new_i_hash in cut_systems.keys():
                    if cut_sys_dict["meta"]["translation"] > cut_systems[new_i_hash][0]:
                        logger.debug("Skipping due to translation")
                        logger.debug(
                            cut_sys_dict["meta"]["translation"],
                            cut_systems[new_i_hash][0],
                        )
                        continue

                cut_systems[new_i_hash] = (
                    cut_sys_dict["meta"]["translation"],
                    cut_sys_dict,
                )

    logger.debug(f"Created {len(cut_systems)} isolated systems.")
    return list(cut_systems.values())


def save_capped_systems(systems, out_dir, frame: int = None):
    """Saves output from `extract_subsystems`

    Parameters
    ----------
    systems : list
        Systems to save the structures and meta file for.
    out_dir : Path
        Where to save. Should probably be traj/batch_238/se
    frame
        Overwrite the frame for all given systems
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for system in systems:
        system = system[1]  # 0 is translation
        sys_hash = f'{system["meta"]["hash_u1"]}_{system["meta"]["hash_u2"]}'

        if (out_dir / f"{sys_hash}.npz").exists():
            # print(f"ERROR: {sys_hash} hash exists!")
            continue

        system["start_u"].atoms.write(out_dir / f"{sys_hash}_1.pdb")
        system["end_u"].atoms.write(out_dir / f"{sys_hash}_2.pdb")

        system["meta"]["meta_path"] = out_dir / f"{sys_hash}.npz"

        if frame is not None:
            system["meta"]["frame"] = frame

        np.savez(out_dir / f"{sys_hash}.npz", system["meta"])