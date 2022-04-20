import numpy as np
from napari.layers import Image, Points
from .. import points_to_labels 


def test_points_to_labels_no_points():
    image_data = np.zeros((10, 10))
    reference_image = Image(image_data)
    points = Points([])
    
    labels_data, labels_state, layer_type = points_to_labels(points, reference_image)
    assert layer_type == 'Labels'
    np.testing.assert_array_equal(labels_data, np.zeros((10, 10)))
    for k, v in labels_state.items():
        if k not in ('name', 'opacity'):
            image_value = getattr(reference_image, k)
            np.testing.assert_equal(v, image_value)


def test_points_to_labels_2d_with_one_pixel_point():
    image_data = np.zeros((10, 10))
    reference_image = Image(image_data)
    points_data = [[4, 6]]
    points = Points(points_data, size=[1])
    
    labels_data, _, _ = points_to_labels(points, reference_image)

    expected_labels_data = np.zeros((10, 10))
    expected_labels_data[4, 6] = 1
    np.testing.assert_array_equal(labels_data, expected_labels_data)


def test_points_to_labels_2d_with_many_pixel_point():
    image_data = np.zeros((10, 10))
    reference_image = Image(image_data)
    points_data = [[4, 6]]
    points = Points(points_data, size=[2])
    
    labels_data, _, _ = points_to_labels(points, reference_image)

    expected_labels_data = np.zeros((10, 10))
    expected_labels_data[4, 5:8] = 1
    expected_labels_data[3:6, 6] = 1
    np.testing.assert_array_equal(labels_data, expected_labels_data)


def test_points_to_labels_2d_with_two_one_pixel_points():
    image_data = np.zeros((10, 10))
    reference_image = Image(image_data)
    points_data = [[4, 6], [7, 5]]
    points = Points(points_data, size=[1])
    
    labels_data, _, _ = points_to_labels(points, reference_image)

    expected_labels_data = np.zeros((10, 10))
    expected_labels_data[4, 6] = 1
    expected_labels_data[7, 5] = 2
    np.testing.assert_array_equal(labels_data, expected_labels_data)
