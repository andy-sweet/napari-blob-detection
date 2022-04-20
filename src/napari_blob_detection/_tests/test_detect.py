import pytest
import numpy as np
from napari.layers import Image
from .. import difference_of_gaussian, laplacian_of_gaussian


METHODS = (difference_of_gaussian, laplacian_of_gaussian)


@pytest.mark.parametrize('method', METHODS)
def test_detect_with_2d_image_no_points(method):
    image = Image(np.zeros((10, 10)))
    
    points_data, points_state, layer_type = method(image, dimensionality=2)

    assert layer_type.lower() == 'points'
    assert points_data.shape == (0, 2)
    assert len(points_state['size']) == 0
    assert len(points_state['features']['sigma']) == 0


@pytest.mark.parametrize('method', METHODS)
def test_detect_with_2d_image_one_2d_point(method):
    image = Image(np.zeros((10, 10)))
    image.data[3:6, 5:8] = 1
    
    points_data, points_state, layer_type = method(image, dimensionality=2)

    assert layer_type.lower() == 'points'
    np.testing.assert_allclose(points_data, [[4, 6]])
    assert len(points_state['size']) == 1
    assert len(points_state['features']['sigma']) == 1
    np.testing.assert_allclose(points_state['size'], np.sqrt(2) * points_state['features']['sigma'])


@pytest.mark.parametrize('method', METHODS)
def test_detect_with_3d_image_two_2d_points(method):
    image = Image(np.zeros((2, 10, 10)))
    image.data[0, 3:6, 5:8] = 1
    image.data[1, 5:8, 3:6] = 1
    
    points_data, points_state, layer_type = method(image, dimensionality=2)

    assert layer_type.lower() == 'points'
    np.testing.assert_allclose(points_data, [[0, 4, 6], [1, 6, 4]])
    assert len(points_state['size']) == 2
    assert len(points_state['features']['sigma']) == 2
    np.testing.assert_allclose(points_state['size'], np.sqrt(2) * points_state['features']['sigma'])


@pytest.mark.parametrize('method', METHODS)
def test_detect_with_3d_image_one_3d_point(method):
    image = Image(np.zeros((10, 10, 10)))
    image.data[4:7, 3:6, 5:8] = 1
    
    points_data, points_state, layer_type = method(image, dimensionality=3)

    assert layer_type.lower() == 'points'
    np.testing.assert_allclose(points_data, [[5, 4, 6]])
    assert len(points_state['size']) == 1
    assert len(points_state['features']['sigma']) == 1
    np.testing.assert_allclose(points_state['size'], np.sqrt(3) * points_state['features']['sigma'])


@pytest.mark.parametrize('method', METHODS)
def test_detect_with_2d_image_3d_features(method):
    image = Image(np.zeros((10, 10)))
    
    with pytest.raises(ValueError):
        method(image, dimensionality=3)
