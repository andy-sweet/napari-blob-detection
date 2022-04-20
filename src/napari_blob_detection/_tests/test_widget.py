import pytest
from .. import detect_blobs_widget
from .. import difference_of_gaussian, laplacian_of_gaussian

from napari.layers import Image
import numpy as np


def test_detect_blobs_widget():
    widget = detect_blobs_widget()
    assert widget is not None
        

@pytest.mark.parametrize('method', [difference_of_gaussian, laplacian_of_gaussian])
def test_detect_feature_points_with_2d_no_points(method):
    image = Image(np.zeros((10, 10)))
    
    points_data, points_state, layer_type = method(image, dimensionality=2)

    assert layer_type.lower() == 'points'
    assert points_data.shape == (0, 2)
    assert len(points_state['size']) == 0
    assert len(points_state['features']['sigma']) == 0


@pytest.mark.parametrize('method', [difference_of_gaussian, laplacian_of_gaussian])
def test_detect_feature_points_with_2d_one_point(method):
    image = Image(np.zeros((10, 10)))
    image.data[4:7, 4:7] = 1
    
    points_data, points_state, layer_type = method(image, dimensionality=2)

    assert layer_type.lower() == 'points'
    np.testing.assert_allclose(points_data, [[5, 5]])
    assert len(points_state['size']) == 1
    assert len(points_state['features']['sigma']) == 1
    np.testing.assert_allclose(points_state['size'], np.sqrt(2) * points_state['features']['sigma'])
