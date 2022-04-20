# convert points layer to labels layer
from skimage.morphology import label

from napari.layers import Image, Points
from napari.types import LayerDataTuple


def points_to_labels(
    points: Points,
    reference_image: Image,
) -> LayerDataTuple:
    """ Converts a points layer to a labels layer.

    Any overlapping points will be assigned the same label value.

    Parameters
    ----------
    points : Points
        The points layer to convert.
    reference_image : Image
        The reference image layer that the points were detected on.
        The shape and transforms of this image will be used for the output
        labels layer.

    Returns
    -------
    LayerDataTuple
        A 3-tuple containing the labels data, other state, and 'Labels'.
    """
    mask_data = points.to_mask(
        shape=reference_image.data.shape,
        data_to_world=reference_image._data_to_world,
        isotropic_output=True,
    )
    data = label(mask_data)
    state = {
        'name': f'{points.name}-labels',
        'scale': reference_image.scale,
        'translate': reference_image.translate,
        'rotate': reference_image.rotate,
        'shear': reference_image.shear,
        'affine': reference_image.affine,
        'opacity': 0.5,
    }
    return data, state, 'Labels'
