from revolve2.core.physics import Terrain
from pyrr import Vector3, Quaternion
from revolve2.core.physics.running import geometry
from typing import Tuple
from noise import pnoise2


def terrain_generator(
    size: Tuple[float, float],
    max_height: float,
    density: float = 1.0,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    import matplotlib.pyplot as plt

    SIZE = 5.0
    NUM_EDGES = 400
    OCTAVE = 10
    SCALE = 20

    num_edges = (
        int(granularity_multiplier * NUM_EDGES * size[0] / SIZE),
        int(granularity_multiplier * NUM_EDGES * size[1] / SIZE),
    )

    if max_height == 0.0:
        heights = [[0.0 for y in range(num_edges[1])] for x in range(num_edges[0])]
        max_height = 1.0
    else:
        heights = [
            [
                pnoise2(
                    x / num_edges[0] * SCALE / SIZE * size[0] * density,
                    y / num_edges[1] * SCALE / SIZE * size[1] * density,
                    OCTAVE,
                )
                for y in range(num_edges[1])
            ]
            for x in range(num_edges[0])
        ]

    plt.imshow(heights, cmap="gray")
    plt.show()

    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3(),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=max_height + 0.1,
                heights=heights,
            )
        ]
    )
