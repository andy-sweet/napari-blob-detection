from skimage.feature import blob_dog


def detect_blobs(
    image: "napari.layers.Image",
    min_sigma: float = 1,
    max_sigma: float = 50,
    threshold: float = 0.5,
) -> "napari.types.PointsData":
    coords = blob_dog(image.data, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    return coords[:, :2]
