import json
import logging

import MDAnalysis as MDA
import numpy as np

from HATfix.utils.trajectory_utils import extract_subsystems, save_capped_systems

from kimmdy.recipe import Bind, Break, Place, Relax, Recipe, RecipeCollection
from kimmdy.plugins import ReactionPlugin

from pprint import pformat
from tempfile import TemporaryDirectory
import shutil
from pathlib import Path
from tqdm.autonotebook import tqdm



class FixedRateHAT(ReactionPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.h_cutoff = self.config.h_cutoff
        self.freqfac = self.config.frequency_factor
        self.barrier = self.config.barrier
        self.polling_rate = self.config.polling_rate

    def get_recipe_collection(self, files) -> RecipeCollection:
        from HATfix.utils.input_generation import metas_to_ds

        logger = files.logger
        logger.debug("Getting recipe for reaction: HATfix")

        tpr = str(files.input["tpr"])
        trr = str(files.input["trr"])
        u = MDA.Universe(str(tpr), str(trr))

        se_dir = files.outputdir / "se"
        if not self.config.keep_structures:
            se_dir_bck = se_dir
            se_tmpdir = TemporaryDirectory()
            se_dir = Path(se_tmpdir.name)

        # One-based strings in top
        rad_ids = [str(int(i) - 1) for i in self.runmng.top.radicals.keys()]
        logger.info(f"HATfix is using radicals from KIMMDY: {rad_ids}")


        if len(rad_ids) < 1:
            logger.info("--> retuning empty recipe collection")
            return RecipeCollection([])
        rad_ids = sorted(rad_ids)
        sub_atms = u.select_atoms(
            f"((not resname SOL NA CL) and (around 20 id {' '.join([i for i in rad_ids])}))"
            f" or id {' '.join([i for i in rad_ids])}",
            updating=True,
        )
        try:
            # environment around radical is updated by ts incrementation
            logger.info("Searching trajectory for radical structures.")
            for ts in tqdm(u.trajectory[:: self.polling_rate]):
                u_sub = MDA.Merge(sub_atms)
                u_sub.trajectory[0].dimensions = ts.dimensions

                # check manually w/ ngl:
                if 0:
                    import nglview as ngl

                    view = ngl.show_mdanalysis(u_sub, defaultRepresentation=False)
                    view.representations = [
                        {"type": "ball+stick", "params": {"sele": ""}},
                        {
                            "type": "spacefill",
                            "params": {"sele": "", "radiusScale": 0.7},
                        },
                    ]
                    view._set_selection("@" + ",".join(rad_ids), repr_index=1)
                    view.center()
                    view

                subsystems = extract_subsystems(
                    u_sub,
                    rad_ids,
                    h_cutoff=self.h_cutoff,
                    env_cutoff=10,
                    start=0,
                    stop=1,
                    step=1,
                    cap=False,
                    rad_min_dist=3,
                    unique=False,
                    logger=logger,
                )
                save_capped_systems(subsystems, se_dir, frame=ts.frame)

            _, _, _, meta_ds, metas_masked = metas_to_ds(
                meta_files=list(se_dir.glob("*.npz")),
                max_dist=None,
                min_dist=None,
                opt=False,
                mask_energy=False,
                oneway=True,
            )

            # Rate; RT=0.593 kcal/mol
            logger.info("Creating Recipes.")
            rate = np.multiply(self.freqfac, np.float_power(np.e, (-self.barrier / 0.593)))
            recipes = []
            logger.debug(f"Barrier:\n{pformat(self.barrier)}")
            logger.info(f"Rate: {rate}, for {len(meta_ds)} possible reactions")
            for meta_d in meta_ds:
                ids = [int(i) for i in meta_d["indices"][0:2]]  # should be zero-based

                f1 = meta_d["frame"]
                f2 = meta_d["frame"] + self.polling_rate
                if f2 >= len(u.trajectory):
                    f2 = len(u.trajectory) - 1
                t1 = u.trajectory[f1].time
                t2 = u.trajectory[f2].time
                old_bound = int(u_sub.select_atoms(f"bonded id {ids[0]}")[0].id)

                # get end position
                pdb_e = meta_d["meta_path"].with_name(
                    meta_d["meta_path"].stem + "_2.pdb"
                )
                with open(pdb_e) as f:
                    finished = False
                    while not finished:
                        line = f.readline()
                        if line[:11] == "ATOM      1":
                            finished = True
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                if self.config.change_coords == "place":
                    # HAT plugin ids are kimmdy ixs (zero-based,int)
                    seq = [
                        Break(old_bound, ids[0]),
                        Place(ix_to_place=ids[0], new_coords=[x, y, z]),
                        Bind(ids[0], ids[1]),
                    ]
                elif self.config.change_coords == "lambda":
                    seq = [Break(old_bound, ids[0]), Bind(ids[0], ids[1]), Relax()]
                else:
                    raise ValueError(
                        f"Unknown change_coords parameter {self.config.change_coords}"
                    )

                # make recipe
                recipes.append(
                    Recipe(recipe_steps=seq, rates=[rate], timespans=[[t1, t2]])
                )

            recipe_collection = RecipeCollection(recipes)
        except Exception as e:
            # backup in case of failure
            if not self.config.keep_structures:
                shutil.copytree(se_dir, se_dir_bck)
            raise e

        if not self.config.keep_structures:
            se_tmpdir.cleanup()
        return recipe_collection
