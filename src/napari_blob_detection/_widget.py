from enum import Enum, auto
from magicgui import magicgui, widgets
from typing import Annotated
from skimage.feature import blob_dog, blob_doh, blob_log
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
    DOH = auto()
    LOG = auto()


ALGORITHM_LOOKUP = {
    Algorithm.DOG: blob_dog,
    Algorithm.DOH: blob_doh,
    Algorithm.LOG: blob_log,
}


def difference_of_gaussian(
    image: 'napari.layers.Image',
    dimensionality: Dimensionality = Dimensionality.TWO_D,
    min_sigma: Annotated[float, {'min': 0.5, 'max': 15, 'step': 0.5}] = 1,
    max_sigma: Annotated[float, {'min': 1, 'max': 1000, 'step': 0.5}] = 50,
    sigma_ratio: Annotated[float, {'min': 1, 'max': 10}] = 1.6,
    threshold: Annotated[float, {'min': 0, 'max': 1000, 'step': 0.1}] = 0.5,
    overlap: Annotated[float, {'min': 0, 'max': 1, 'step': 0.01}] = 0.5,
    exclude_border: bool = False,
) -> 'napari.types.LayerDataTuple':
    kwargs = locals()
    return detect_blobs(
        kwargs.pop('image'),
        Algorithm.DOG,
        kwargs.pop('dimensionality'),
        **kwargs,
    )


def determinant_of_hessian(
    image: 'napari.layers.Image',
    min_sigma: Annotated[float, {'min': 0.5, 'max': 15, 'step': 0.5}] = 1,
    max_sigma: Annotated[float, {'min': 1, 'max': 1000, 'step': 0.5}] = 30,
    num_sigma: Annotated[int, {'min': 1, 'max': 20}] = 10,
    threshold: Annotated[float, {'min': 0, 'max': 1000, 'step': 0.01}] = 0.01,
    overlap: Annotated[float, {'min': 0, 'max': 1, 'step': 0.01}] = 0.5,
    log_scale: bool = False,
) -> 'napari.types.LayerDataTuple':
    kwargs = locals()
    return detect_blobs(
        kwargs.pop('image'),
        Algorithm.DOH,
        Dimensionality.TWO_D,
        **kwargs,
    )


def laplacian_of_gaussian(
    image: 'napari.layers.Image',
    dimensionality: Dimensionality = Dimensionality.TWO_D,
    min_sigma: Annotated[float, {'min': 0.5, 'max': 15, 'step': 0.5}] = 1,
    max_sigma: Annotated[float, {'min': 1, 'max': 1000, 'step': 0.5}] = 50,
    num_sigma: Annotated[int, {'min': 1, 'max': 20}] = 10,
    threshold: Annotated[float, {'min': 0, 'max': 1000, 'step': 0.1}] = 0.2,
    overlap: Annotated[float, {'min': 0, 'max': 1, 'step': 0.01}] = 0.5,
    log_scale: bool = False,
    exclude_border: bool = False,
) -> 'napari.types.LayerDataTuple':
    kwargs = locals()
    return detect_blobs(
        kwargs.pop('image'),
        Algorithm.LOG,
        kwargs.pop('dimensionality'),
        **kwargs,
    )


def detect_blobs(
    image: 'napari.layers.Image',
    algorithm: Algorithm = Algorithm.DOG,
    dimensionality: Dimensionality = Dimensionality.TWO_D,
    **kwargs,
) -> 'napari.types.LayerDataTuple':
    data = image.data
    all_coords = []
    all_sigmas = []
    algorithm_func = ALGORITHM_LOOKUP[algorithm]
    ndim = DIMENSIONALITY_LOOKUP[dimensionality]
    last_dims_index = tuple(slice(n) for n in data.shape[-ndim:])
    for index in np.ndindex(data.shape[:-ndim]):
        full_index = index + last_dims_index
        indexed_image_data = data[full_index]
        coords = algorithm_func(indexed_image_data, **kwargs)
        for c in coords:
            all_coords.append(index + tuple(c[:ndim]))
            all_sigmas.append(c[-1])
    state = {
        'name': f'{image.name}-{algorithm}',
        'face_color': 'red',
        'opacity': 0.5,
        'features': {'sigma': all_sigmas},
        'scale': image.scale,
        'translate': image.translate,
        'rotate': image.rotate,
        'shear': image.shear,
        'affine': image.affine, 
        'size': np.sqrt(ndim) * np.array(all_sigmas),
    }
    return (all_coords, state, 'Points')


METHODS = {
    'Difference of Gaussian': difference_of_gaussian,
    'Determinant of Hessian': determinant_of_hessian,
    'Laplacian of Gaussian': laplacian_of_gaussian,
}


def detect_blobs_widget() -> widgets.Container:
    # make a widget function that will select from the methods
    method = widgets.ComboBox(choices=tuple(METHODS), name='method')
    container = widgets.Container(widgets=[method], labels=False)

    # when the method changes, populate the container with the correct widget
    @method.changed.connect
    def _add_subwidget(method_name: str):
        if len(container) > 1:
            container.pop(-1).native.close()
        subwidget = magicgui(METHODS[method_name])
        subwidget.margins = (0, 0, 0, 0)
        container.append(subwidget)

    method.value = 'Difference of Gaussian'
    _add_subwidget(method.value)

    return container
