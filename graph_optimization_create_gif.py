import numpy as np
import matplotlib.pyplot as plt
from graph_generalist_optimizer import (
    DbGraphGeneralistOptimizerGraphNodeState,
    DbGenotype,
)
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import os
import PIL
import random
import copy
import subprocess


def main() -> None:
    bodies = ["body1", "body2", "body3", "body4", "body5"]
    envs = ["flat plane"]

    dbname = "graph_generalist_run0"
    figsdir = f"./{dbname}_figs"

    db = open_database_sqlite(dbname)
    process_id = 0

    df = pandas.read_sql(
        select(DbGenotype, DbGraphGeneralistOptimizerGraphNodeState).filter(
            (DbGraphGeneralistOptimizerGraphNodeState.process_id == process_id)
            & (DbGraphGeneralistOptimizerGraphNodeState.genotype_id == DbGenotype.id)
        ),
        db,
    )
    min_id = df.genotype_id.min()
    max_id = df.genotype_id.max()

    colormap = {id + min_id: id for id in range(max_id - min_id + 1)}
    shuffledvalues = list(colormap.values())
    random.shuffle(shuffledvalues)
    colormap = dict(zip(colormap, shuffledvalues))

    os.makedirs(figsdir)

    for gen_i in range(len(df.gen_num.value_counts())):
        gen = df[df.gen_num == gen_i]

        ids = np.array([gen.genotype_id])
        reassigned_colors = ids.copy()

        for i in range(len(ids)):
            for j in range(len(ids[i])):
                reassigned_colors[i][j] = colormap[ids[i][j]]

        fig, ax = plt.subplots()
        ax.imshow(reassigned_colors, vmin=min_id, vmax=max_id)

        ax.set_xticks(np.arange(len(bodies)), labels=bodies)
        ax.set_yticks(np.arange(len(envs)), labels=envs)

        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(envs)):
            for j in range(len(bodies)):
                text = ax.text(j, i, ids[i, j], ha="center", va="center", color="w")

        ax.set_title(f"Graph optimizer\nGenotype for each node\nGeneration {gen_i:05d}")
        fig.tight_layout()
        fig.savefig(os.path.join(figsdir, f"gen_{gen_i:05d}.png"))
        plt.close(fig)

    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            "20",
            "-pattern_type",
            "glob",
            "-i",
            f"{figsdir}/*.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            f"{figsdir}/evolution.mp4",
        ]
    )


if __name__ == "__main__":
    main()
