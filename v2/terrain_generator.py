from revolve2.core.physics import Terrain
from pyrr import Vector3, Quaternion
from revolve2.core.physics.running import geometry
from typing import Tuple, List
from noise import pnoise2
import math
import numpy as np
import numpy.typing as npt


def terrain_generator(
    size: Tuple[float, float],
    ruggedness: float,
    bowlness: float,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    NUM_EDGES = 100

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    bowl = bowl_heighmap(num_edges=num_edges)

    max_height = ruggedness + bowlness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
    else:
        heightmap = (ruggedness * rugged + bowlness * bowl) / (ruggedness + bowlness)

    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3(),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            )
        ]
    )


def rugged_heightmap(
    size: Tuple[float, float],
    num_edges: Tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    # TODO the maximum height is not really 1.0.
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: pnoise2(
                x / num_edges[0] * C1 * size[0] * density,
                y / num_edges[1] * C1 * size[1] * density,
                OCTAVE,
            ),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )


def bowl_heighmap(
    num_edges: Tuple[int, int],
) -> List[List[float]]:
    return np.fromfunction(
        np.vectorize(
            lambda y, x: (x / num_edges[0] * 2.0 - 1.0) ** 2
            + (y / num_edges[1] * 2.0 - 1.0) ** 2
            if math.sqrt(
                (x / num_edges[0] * 2.0 - 1.0) ** 2
                + (y / num_edges[1] * 2.0 - 1.0) ** 2
            )
            <= 1.0
            else 0.0,
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )
