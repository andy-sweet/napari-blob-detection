from magicgui import magicgui, widgets
from typing import Annotated, Callable
from skimage.feature import blob_dog, blob_doh, blob_log
import numpy as np


def difference_of_gaussian(
    image: 'napari.layers.Image',
    dimensionality: Annotated[int, {'choices': [2, 3]}] = 2,
    min_sigma: Annotated[float, {'min': 0.5, 'max': 15, 'step': 0.5}] = 1,
    max_sigma: Annotated[float, {'min': 1, 'max': 1000, 'step': 0.5}] = 50,
    sigma_ratio: Annotated[float, {'min': 1, 'max': 10}] = 1.6,
    threshold: Annotated[float, {'min': 0, 'max': 1000, 'step': 0.1}] = 0.5,
    overlap: Annotated[float, {'min': 0, 'max': 1, 'step': 0.01}] = 0.5,
    exclude_border: bool = False,
) -> 'napari.types.LayerDataTuple':
    kwargs = locals()
    return _detect_blobs(
        image=kwargs.pop('image'),
        method=blob_dog,
        dimensionality=kwargs.pop('dimensionality'),
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
    return _detect_blobs(
        image=kwargs.pop('image'),
        method=blob_doh,
        dimensionality=2,
        **kwargs,
    )


def laplacian_of_gaussian(
    image: 'napari.layers.Image',
    dimensionality: Annotated[int, {'choices': [2, 3]}] = 2,
    min_sigma: Annotated[float, {'min': 0.5, 'max': 15, 'step': 0.5}] = 1,
    max_sigma: Annotated[float, {'min': 1, 'max': 1000, 'step': 0.5}] = 50,
    num_sigma: Annotated[int, {'min': 1, 'max': 20}] = 10,
    threshold: Annotated[float, {'min': 0, 'max': 1000, 'step': 0.1}] = 0.2,
    overlap: Annotated[float, {'min': 0, 'max': 1, 'step': 0.01}] = 0.5,
    log_scale: bool = False,
    exclude_border: bool = False,
) -> 'napari.types.LayerDataTuple':
    kwargs = locals()
    return _detect_blobs(
        image=kwargs.pop('image'),
        method=blob_log,
        dimensionality=kwargs.pop('dimensionality'),
        **kwargs,
    )


def _detect_blobs(
    *,
    image: 'napari.layers.Image',
    method: Callable[..., np.ndarray],
    dimensionality: Annotated[int, {'choices': [2, 3]}] = 2,
    **kwargs,
) -> 'napari.types.LayerDataTuple':
    data = image.data
    if data.ndim < dimensionality:
        raise ValueError(f'The input image has fewer dimensions ({data.ndim}) than the feature dimensionality ({dimensionality})')
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
    state = {
        'name': f'{image.name}-features-{method.__name__}',
        'face_color': 'red',
        'opacity': 0.5,
        'features': {'sigma': all_sigmas},
        'scale': image.scale,
        'translate': image.translate,
        'rotate': image.rotate,
        'shear': image.shear,
        'affine': image.affine, 
        'size': np.sqrt(dimensionality) * np.array(all_sigmas),
    }
    return (all_coords, state, 'Points')


METHODS = {
    'Difference of Gaussian': difference_of_gaussian,
    'Determinant of Hessian': determinant_of_hessian,
    'Laplacian of Gaussian': laplacian_of_gaussian,
}


def detect_blobs_widget() -> widgets.Container:
    # Make a widget function that will select from the methods.
    method = widgets.ComboBox(choices=tuple(METHODS), name='method')
    container = widgets.Container(widgets=[method], labels=False)

    # When the method changes, populate the container with the correct widget.
    @method.changed.connect
    def _add_subwidget(method_name: str):
        if len(container) > 1:
            container.pop(-1).native.close()
        subwidget = magicgui(METHODS[method_name])
        subwidget.margins = (0, 0, 0, 0)
        container.append(subwidget)

    # Set the default method and force the subwidget to update accordingly.
    method.value = 'Difference of Gaussian'
    _add_subwidget(method.value)

    return container
