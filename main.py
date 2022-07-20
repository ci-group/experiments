from revolve2.core.modular_robot import Body


def learn_reduced_bias_develop() -> list[float]:
    """
    Learns parameters to pass to develop so it becomes less biased
    """
    pass


def develop(genotype: list[float], params: list[float]) -> Body:
    pass


def similarity(body1: Body, body2: Body) -> float:
    pass
