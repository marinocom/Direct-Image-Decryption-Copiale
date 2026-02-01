"""Input / Output miscellaneous operations."""

import pickle
from pathlib import Path
from typing import Any, Dict


def load_pickle_prediction(path: Path) -> Dict[str, Any]:
    """Load a prediction pickle file into memory.

    Parameters
    ----------
    path: Path
        Path to the prediction file.

    Returns
    -------
    Dict[str, Any]
        A dictionary whose keys are metric fields and values are the stored numpy
        arrays.
    """
    obj = []
    with open(path, "rb") as f_in:
        while 1:
            try:
                obj.append(pickle.load(f_in))
            except EOFError:
                break
    return {k: v for x in obj for k, v in x.items()}
