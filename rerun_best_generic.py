from typing import List
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from revolve2.core.optimization.ea.openai_es import DbOpenaiESOptimizerIndividual
import math
from revolve2.core.database.serializers import (
    Ndarray1xnSerializer,
)
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from dof_map_brain import DofMapBrain
from revolve2.runners.isaacgym import ModularRobotRerunner


async def rerun_best(
    database_name: str,
    bodies: List[Body],
    dof_maps: List[List[int]],
    cpg_network_structure: CpgNetworkStructure,
) -> None:
    db = open_async_database_sqlite(database_name)
    async with AsyncSession(db) as session:
        best_individual = (
            (
                await session.execute(
                    select(DbOpenaiESOptimizerIndividual).order_by(
                        DbOpenaiESOptimizerIndividual.fitness.desc()
                    )
                )
            )
            .scalars()
            .all()[0]
        )

        params = [
            p
            for p in (
                await Ndarray1xnSerializer.from_database(
                    session, [best_individual.individual]
                )
            )[0]
        ]

        print(f"fitness: {best_individual.fitness}")
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
    await rerunner.rerun(robots, 10)
