from magicgui.widgets import ComboBox, Container, FunctionGui

from .. import detect_blobs_widget


def test_detect_blobs_widget():
    widget = detect_blobs_widget()

    assert isinstance(widget, Container)
    assert isinstance(widget.method, ComboBox)
    assert isinstance(widget.difference_of_gaussian, FunctionGui)


def test_change_method():
    widget = detect_blobs_widget()
    widget.method.value = 'Laplacian of Gaussian'
    assert isinstance(widget.laplacian_of_gaussian, FunctionGui)
