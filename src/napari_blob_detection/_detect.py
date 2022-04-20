from typing import Callable
from typing_extensions import Annotated
from skimage.feature import blob_dog, blob_log
import numpy as np
from napari.layers import Image
from napari.types import LayerDataTuple

# Define common argument types.
Dimensionality = Annotated[int, {'choices': [2, 3]}]
MinSigma = Annotated[float, {'min': 0.5, 'max': 15, 'step': 0.5}]
MaxSigma = Annotated[float, {'min': 1, 'max': 1000, 'step': 0.5}]
Threshold = Annotated[float, {'min': 0, 'max': 1000, 'step': 0.1}]


def difference_of_gaussian(
    image: Image,
    *,
    dimensionality: Dimensionality = 2,
    min_sigma: MinSigma = 1,
    max_sigma: MaxSigma = 50,
    threshold: Threshold = 0.5,
) -> LayerDataTuple:
    """ Detects features points on an image layer using the Difference of Gaussian method.

    The dimensionality of the image must be at least as high as the dimensionality of the
    features. If the image has more dimensions than the features, this will iterate over
    the leading extra dimensions.

    Parameters
    ----------
    image : Image
        Image layer for blob detection. Can be a 2D, 3D, or higher dimensionality image.
    dimensionality : Literal[2, 3]
        The dimensionality of the blobs to find.
    min_sigma : float
        The smallest blob size to detect.
    max_sigma : float
        The largest blob size to detect.
    threshold : float
        Reduce this to detect blobs with lower intensities.

    Returns
    -------
    LayerDataTuple
        A 3-tuple containing the feature points data, other state, and 'Points'.
    """
    kwargs = locals()
    return _detect_blobs(
        image=kwargs.pop('image'),
        method=blob_dog,
        dimensionality=kwargs.pop('dimensionality'),
        **kwargs,
    )


def laplacian_of_gaussian(
    image: Image,
    *,
    dimensionality: Dimensionality = 2,
    min_sigma: MinSigma = 1,
    max_sigma: MaxSigma = 50,
    threshold: Threshold = 0.5,
) -> LayerDataTuple:
    """ Detects features points on an image layer.

    The dimensionality of the image must be at least as high as the dimensionality of the
    features. If the image has more dimensions than the features, this will iterate over
    the leading extra dimensions.

    Parameters
    ----------
    image : Image
        Image layer for blob detection. Can be a 2D, 3D, or higher dimensionality image.
    dimensionality : Literal[2, 3]
        The dimensionality of the blobs to find.
    min_sigma : float
        The smallest blob size to detect.
    max_sigma : float
        The largest blob size to detect.
    threshold : float
        Reduce this to detect blobs with lower intensities.

    Returns
    -------
    LayerDataTuple
        A 3-tuple containing the feature points data, other state, and 'Points'.
    """
    kwargs = locals()
    return _detect_blobs(
        image=kwargs.pop('image'),
        method=blob_log,
        dimensionality=kwargs.pop('dimensionality'),
        **kwargs,
    )


def _detect_blobs(
    *,
    image: Image,
    method: Callable[..., np.ndarray],
    dimensionality: Dimensionality = 2,
    **kwargs,
) -> LayerDataTuple:
    data = image.data
    if data.ndim < dimensionality:
        raise ValueError(f'The input image has fewer dimensions ({data.ndim}) than the feature dimensionality ({dimensionality})')
    # Find features in the last dimensions of the image and iterate over
    # leading dimensions.
    feature_slices = tuple(slice(n) for n in data.shape[-dimensionality:])
    all_coords = []
    all_sigmas = []
    for index in np.ndindex(data.shape[:-dimensionality]):
        full_index = index + feature_slices
        indexed_image_data = data[full_index]
        coords = method(indexed_image_data, **kwargs)
        for c in coords:
            all_coords.append(index + tuple(c[:-1]))
            all_sigmas.append(c[-1])
    all_coords = np.reshape(all_coords, (-1, image.ndim))
    all_sigmas = np.array(all_sigmas)
    state = {
        'name': f'{image.name}-features-{method.__name__}',
        'features': {'sigma': all_sigmas},
        'scale': image.scale,
        'translate': image.translate,
        'rotate': image.rotate,
        'shear': image.shear,
        'affine': image.affine, 
        'opacity': 0.5,
        'face_color': 'red',
        'size': np.sqrt(dimensionality) * all_sigmas,
    }
    return (all_coords, state, 'Points')
