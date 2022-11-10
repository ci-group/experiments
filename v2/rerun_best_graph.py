from bodies import make_bodies, make_cpg_network_structure

import argparse
from revolve2.core.database import open_async_database_sqlite, open_database_sqlite
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
import graph_optimizer
import math
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from dof_map_brain import DofMapBrain
from revolve2.runners.mujoco import ModularRobotRerunner
from revolve2.core.modular_robot import ModularRobot
import pandas
import numpy as np
from revolve2.core.physics.running import RecordSettings


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to fetch brain parameters from.",
    )
    parser.add_argument(
        "body",
        type=int,
        help="Number of the body to simulate.",
    )
    parser.add_argument(
        "-r",
        "--record_to_directory",
        type=str,
        help="If set, videos are recorded and stored in this directory.",
    )
    args = parser.parse_args()

    db = open_database_sqlite(args.database)
    df = pandas.read_sql(
        select(
            graph_optimizer.ProgramState.table,
            graph_optimizer.Population.item_table,
            graph_optimizer.Measures.table,
        ).filter(
            (
                graph_optimizer.ProgramState.table.population
                == graph_optimizer.Population.item_table.list_id
            )
            & (
                graph_optimizer.Population.item_table.measures
                == graph_optimizer.Measures.table.id
            )
            & (graph_optimizer.Population.item_table.index == args.body)
        ),
        db,
    )
    bestrow = df[df.fitness == df.fitness.max()]
    bestrow = bestrow[bestrow.generation_index == bestrow.generation_index.max()]
    best_params_id = int(bestrow.genotype)

    db2 = open_async_database_sqlite(args.database)
    async with AsyncSession(db2) as session:
        best_params = await graph_optimizer.Parameters.from_db(session, best_params_id)

    print(f"fitness: {float(bestrow.fitness)}")
    print(f"params: {best_params}")

    bodies, dof_maps = make_bodies()
    bodies = [bodies[args.body]]
    dof_maps = [dof_maps[args.body]]
    cpg_network_structure = make_cpg_network_structure()

    robots = []
    for body, dof_map in zip(bodies, dof_maps):
        weight_matrix = (
            cpg_network_structure.make_connection_weights_matrix_from_params(
                np.clip(best_params, 0.0, 1.0) * 4.0 - 2.0
            )
        )
        initial_state = cpg_network_structure.make_uniform_state(math.sqrt(2) / 2.0)
        dof_ranges = cpg_network_structure.make_uniform_dof_ranges(1.0)

        inner_brain = BrainCpgNetworkStatic(
            initial_state,
            cpg_network_structure.num_cpgs,
            weight_matrix,
            dof_ranges,
        )
        brain = DofMapBrain(inner_brain, dof_map)
        robots.append(ModularRobot(body, brain))

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(
        robots,
        60,
        simulation_time=30,
        start_paused=False,
        record_settings=None
        if args.record_to_directory is None
        else RecordSettings(video_directory=args.record_to_directory),
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
