from bodies import make_bodies, make_cpg_network_structure

from rerun_best_generic import rerun_best
import argparse


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to fetch brain parameters from.",
    )
    # args = parser.parse_args()

    bodies, dof_maps = make_bodies()
    cpg_network_structure = make_cpg_network_structure()

    await rerun_best(
        database_name=123,
        bodies=bodies,
        dof_maps=dof_maps,
        cpg_network_structure=cpg_network_structure,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
