from typing import Optional
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.database import open_async_database_sqlite, open_database_sqlite
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
import de_multi_body_optimizer
import math
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from dof_map_brain import DofMapBrain
from revolve2.runners.mujoco import ModularRobotRerunner
import numpy as np
from bodies import make_bodies, make_cpg_network_structure
import pandas
from revolve2.core.physics.running import RecordSettings


async def rerun_best(
    database_name: str, body: int, record_to_directory: Optional[str]
) -> None:
    db = open_database_sqlite(database_name)
    df = pandas.read_sql(
        select(
            de_multi_body_optimizer.ProgramState.table,
            de_multi_body_optimizer.Population.item_table,
            de_multi_body_optimizer.Measures.table,
        ).filter(
            (
                de_multi_body_optimizer.ProgramState.table.population
                == de_multi_body_optimizer.Population.item_table.list_id
            )
            & (
                de_multi_body_optimizer.Population.item_table.measures
                == de_multi_body_optimizer.Measures.table.id
            )
        ),
        db,
    )
    bestrow = df[df.fitness == df.fitness.max()]
    bestrow = bestrow[bestrow.generation_index == bestrow.generation_index.max()]
    best_params_id = int(bestrow.genotype)

    db2 = open_async_database_sqlite(database_name)
    async with AsyncSession(db2) as session:
        best_params = await de_multi_body_optimizer.Genotype.from_db(
            session, best_params_id
        )

    print(f"fitness: {float(bestrow.fitness)}")
    print(f"params: {best_params}")

    bodies, dof_maps = make_bodies()
    bodies = [bodies[body]]
    dof_maps = [dof_maps[body]]
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
        if record_to_directory is None
        else RecordSettings(video_directory=record_to_directory, fps=60),
    )
