from enum import Enum, auto
from skimage.feature import blob_dog, blob_doh, blob_log
import numpy as np


class Dimensionality(Enum):
    TWO_D = auto()
    THREE_D = auto()


class Algorithm(Enum):
    DOG = auto()
    DOH = auto()
    LOG = auto()


ALGORITHM_LOOKUP = {
    Algorithm.DOG: blob_dog,
    Algorithm.DOH: blob_doh,
    Algorithm.LOG: blob_log,
}


def detect_blobs(
    image: "napari.layers.Image",
    min_sigma: float = 1,
    max_sigma: float = 50,
    threshold: float = 0.5,
    dimensionality: Dimensionality = Dimensionality.TWO_D,
    algorithm: Algorithm = Algorithm.DOG,
) -> "napari.types.LayerDataTuple":
    data = image.data
    all_coords = []
    all_sigmas = []
    # TODO: only works for 3D data, understand to generalize this.
    algorithm_func = ALGORITHM_LOOKUP[algorithm]
    if dimensionality == Dimensionality.TWO_D:
        for i in range(data.shape[0]):
            coords = algorithm_func(data[i, :, :], min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
            for c in coords:
                all_coords.append([i] + list(c[:2]))
                all_sigmas.append(c[-1])
    state = {
        'name': f'{image.name}-{algorithm}',
        'face_color': 'red',
        'opacity': 0.5,
        'features': {'sigma': all_sigmas},
        'size': np.sqrt(2) * np.array(all_sigmas),
    }
    return (all_coords, state, 'Points')



