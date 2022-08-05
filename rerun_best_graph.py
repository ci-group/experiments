from bodies import make_bodies, make_cpg_network_structure

import argparse
from revolve2.core.database import open_async_database_sqlite, open_database_sqlite
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from graph_generalist_optimizer import (
    DbGraphGeneralistOptimizerGraphNodeState,
    DbGenotype,
)
from revolve2.core.database.serializers import (
    Ndarray1xnSerializer,
)
import math
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from dof_map_brain import DofMapBrain
from revolve2.runners.isaacgym import ModularRobotRerunner
from revolve2.core.modular_robot import ModularRobot
import pandas


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
    args = parser.parse_args()

    bodies, dof_maps = make_bodies()
    bodies = [bodies[args.body]]
    dof_maps = [dof_maps[args.body]]
    cpg_network_structure = make_cpg_network_structure()

    db = open_database_sqlite(args.database)
    df = pandas.read_sql(
        select(DbGraphGeneralistOptimizerGraphNodeState, DbGenotype).filter(
            (DbGraphGeneralistOptimizerGraphNodeState.graph_index == args.body)
            & (DbGraphGeneralistOptimizerGraphNodeState.genotype_id == DbGenotype.id)
        ),
        db,
    )
    bestrow = df[df.fitness == df.fitness.max()]
    bestrow = bestrow[df.gen_num == df.gen_num.max()]
    bestarrayid = int(bestrow.nparray1xn_id)

    db2 = open_async_database_sqlite(args.database)
    async with AsyncSession(db2) as session:
        params = [
            p
            for p in (await Ndarray1xnSerializer.from_database(session, [bestarrayid]))[
                0
            ]
        ]

        print(f"fitness: {bestrow.fitness}")
        print(f"params: {params}")

        robots = []
        for body, dof_map in zip(bodies, dof_maps):
            weight_matrix = (
                cpg_network_structure.make_connection_weights_matrix_from_params(params)
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
    await rerunner.rerun(robots, 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
