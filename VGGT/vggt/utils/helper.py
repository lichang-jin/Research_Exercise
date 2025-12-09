import numpy as np


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values, randomly keep only max_trues of them and set the rest to False.
    :param mask:
    :param max_trues:
    :return:
    """