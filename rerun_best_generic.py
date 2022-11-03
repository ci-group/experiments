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
import numpy as np


async def rerun_best(
    database_name: str,
    bodies: List[Body],
    dof_maps: List[List[int]],
    cpg_network_structure: CpgNetworkStructure,
) -> None:
    # db = open_async_database_sqlite(database_name)
    # async with AsyncSession(db) as session:
    # best_individual = (
    #     (
    #         await session.execute(
    #             select(DbOpenaiESOptimizerIndividual).order_by(
    #                 DbOpenaiESOptimizerIndividual.fitness.desc()
    #             )
    #         )
    #     )
    #     .scalars()
    #     .all()[0]
    # )

    # params = [
    #     p
    #     for p in (
    #         await Ndarray1xnSerializer.from_database(
    #             session, [best_individual.individual]
    #         )
    #     )[0]
    # ]

    # print(f"fitness: {best_individual.fitness}")
    # print(f"params: {params}")

    params = [
        -0.135853601426204,
        0.507469860544527,
        0.78966573245987,
        0.0927454755233807,
        0.578758503323503,
        0.657062821959638,
        0.485557769998864,
        0.760313803891099,
        0.98869533336782,
        0.182943324675719,
        1.26415306753722,
        0.130604291045897,
        -0.55265648322164,
        1.34413774200747,
        0.957403377846173,
        0.65512106399138,
        0.913690762707389,
        0.63780579312536,
        1.052438455368,
        0.381814779966239,
        -0.36734470559403,
    ]

    robots = []
    for body, dof_map in zip(bodies, dof_maps):
        weight_matrix = (
            cpg_network_structure.make_connection_weights_matrix_from_params(
                np.clip(params, 0.0, 1.0) * 4.0 - 2.0
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
    await rerunner.rerun(robots, 60)
