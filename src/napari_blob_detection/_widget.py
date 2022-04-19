from enum import Enum, auto
from skimage.feature import blob_dog, blob_log
import numpy as np


class Dimensionality(Enum):
    TWO_D = auto()
    THREE_D = auto()


DIMENSIONALITY_LOOKUP = {
    Dimensionality.TWO_D: 2,
    Dimensionality.THREE_D: 3,
}

class Algorithm(Enum):
    DOG = auto()
    LOG = auto()


ALGORITHM_LOOKUP = {
    Algorithm.DOG: blob_dog,
    Algorithm.LOG: blob_log,
}


def detect_blobs(
    image: "napari.layers.Image",
    min_sigma: float = 1,
    max_sigma: float = 10,
    threshold: float = 0.1,
    dimensionality: Dimensionality = Dimensionality.TWO_D,
    algorithm: Algorithm = Algorithm.DOG,
) -> "napari.types.LayerDataTuple":

    """ Detects features points on an image layer.
    
    Parameters
    ----------
    image : napari.layers.Image
        Image layer for blob detection. Can be a 2D, 3D, or higher dimensionality image.
    min_sigma : float
        The smallest blob size to detect.
    max_sigma : float
        The largest blob size to detect.
    threshold : float
        Reduce this to detect blobs with lower intensities.
    dimensionality: 
        Specify if the image is 2D(+t) or 3D(+t).
    algorithm:
        LOG is the most accurate and slowest approach.
        DOG is a faster approximation of LOG approach.
    """

    data = image.data
    all_coords = []
    all_sigmas = []
    algorithm_func = ALGORITHM_LOOKUP[algorithm]
    ndim = DIMENSIONALITY_LOOKUP[dimensionality]
    if ndim == 2:
        radius_factor = np.sqrt(2)
    else:
        radius_factor = np.sqrt(3)

    last_dims_index = tuple(slice(n) for n in data.shape[-ndim:])
    for index in np.ndindex(data.shape[:-ndim]):
        full_index = index + last_dims_index
        indexed_image_data = data[full_index]
        coords = algorithm_func(indexed_image_data, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        for c in coords:
            all_coords.append(index + tuple(c[:ndim]))
            all_sigmas.append(c[-1])
    state = {
        'name': f'{image.name}-{algorithm}',
        'face_color': 'red',
        'opacity': 0.5,
        'features': {'radius': radius_factor * np.array(all_sigmas)},
        'size': radius_factor * np.array(all_sigmas),
        # match the scale of the image and the point
        # point coordinates are not affected 
        'scale': image.scale,
    }
    return (all_coords, state, 'Points')



