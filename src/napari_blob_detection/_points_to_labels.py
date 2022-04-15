# convert points layer to labels layer
from skimage.morphology import label

def points_to_labels(
    point: "napari.layers.Points",
    image: "napari.layers.Image",
) -> "napari.types.LabelsData":
    # points to labels
    mask = point.to_mask(shape=image.data.shape)
    return label(mask)