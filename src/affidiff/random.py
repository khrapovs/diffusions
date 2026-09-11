from numpy.random import Generator, default_rng


def get_random_generator(seed: int | None = None) -> Generator:
    """Get numpy random number generator.

    Parameters
    ----------
    seed
        Random seed

    Returns
    -------
    Generator
    """
    return default_rng(seed=seed)
