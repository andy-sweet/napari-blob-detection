import pytest
from .. import detect_blobs_widget
from .. import difference_of_gaussian, laplacian_of_gaussian

from napari.layers import Image
import numpy as np


def test_detect_blobs_widget():
    widget = detect_blobs_widget()
    assert widget is not None
        

@pytest.mark.parametrize('method', [difference_of_gaussian, laplacian_of_gaussian])
def test_difference_of_gaussian(method):
    image = Image(np.zeros((6, 5)))
    
    points_data, points_state, layer_type = method(image, dimensionality=2)

    assert layer_type.lower() == 'points'
    np.testing.assert_array_equal(points_data, np.empty((0, 2), dtype=float))
    np.testing.assert_array_equal(points_state['size'], [])
    np.testing.assert_array_equal(points_state['features']['sigma'], [])
