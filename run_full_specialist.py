from full_gen_spec import run_full_gen_spec
from bodies import make_bodies, make_cpg_network_structure
from revolve2.core.modular_robot import Body
from typing import List


async def main() -> None:
    await run_all_full_specialist_runs()  # For the actual experiments
    # await dbg_run_full_specialist()  # For debugging only


async def dbg_run_full_specialist() -> None:
    bodies, dof_maps = make_bodies()
    body_num = 2
    await run_full_specialist(
        database_name=f"dbg_full_specialist_body{body_num}",
        headless=True,
        rng_seed=0,
        body=bodies[body_num],
        dof_map=dof_maps[body_num],
    )


async def run_all_full_specialist_runs() -> None:
    NUM_RUNS = 20
    bodies, dof_maps = make_bodies()
    for (sigma, learning_rate) in [(0.5, 0.1), (0.05, 0.01)]:
        for i_body, (body, dof_map) in enumerate(zip(bodies, dof_maps)):
            for i_run in range(NUM_RUNS):
                await run_full_specialist(
                    database_name=f"full_specialist_s{sigma}l{learning_rate}_body{i_body}_run{i_run}",
                    headless=True,
                    rng_seed=10000 * i_body + i_run,
                    body=body,
                    dof_map=dof_map,
                    sigma=sigma,
                    learning_rate=learning_rate,
                )


async def run_full_specialist(
    database_name: str,
    headless: bool,
    rng_seed: int,
    body: Body,
    dof_map: List[int],
    sigma: float,
    learning_rate: float,
) -> None:
    num_evaluations = 10000 / len(make_bodies()[0])
    cpg_network_structure = make_cpg_network_structure()

    await run_full_gen_spec(
        database_name=database_name,
        bodies=[body],
        dof_maps=[dof_map],
        cpg_network_structure=cpg_network_structure,
        headless=headless,
        rng_seed=rng_seed,
        num_evaluations=num_evaluations,
        sigma=sigma,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
