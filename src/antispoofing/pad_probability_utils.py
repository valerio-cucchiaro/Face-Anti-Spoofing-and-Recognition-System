"""
Helpers for interpreting PAD classifier probability columns.

PAD labels use the project convention:
- 0 -> spoof
- 1 -> live

Scikit-learn exposes probability columns in ``classifier.classes_`` order,
so callers must not assume that column 1 is always the live probability.
"""

import numpy as np


PAD_SPOOF_LABEL = 0
PAD_LIVE_LABEL = 1


def get_pad_class_probability_indices(classifier):
    """
    Return probability-column indices for spoof and live classes.

    A missing or incompatible ``classes_`` attribute means the model
    artifact cannot be interpreted safely for PAD metrics/runtime.
    """

    classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise ValueError("PAD classifier does not expose classes_.")

    classes_array = np.asarray(classes).reshape(-1)

    def _index_for(label):
        matches = np.where(classes_array == label)[0]
        if matches.size != 1:
            raise ValueError(
                f"PAD classifier classes_ must contain label {label!r} exactly once; "
                f"found classes_={classes_array.tolist()}."
            )
        return int(matches[0])

    return {
        "spoof": _index_for(PAD_SPOOF_LABEL),
        "live": _index_for(PAD_LIVE_LABEL),
    }
