import numpy as np


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """If mask has more than max_trues True values, randomly keep only max_trues of them and set the rest to False."""
    true_index = np.flatnonzero(mask)
    if len(true_index) <= max_trues:
        return mask

    sampled_index = np.random.choice(true_index, size=max_trues, replace=False)
    limit_flat_mask = np.zeros(mask.size, dtype=bool)
    limit_flat_mask[sampled_index] = True
    return limit_flat_mask.reshape(mask.shape)