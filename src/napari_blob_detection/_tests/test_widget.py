from .._widget import detect_blobs_widget


def test_detect_blobs_widget():
    widget = detect_blobs_widget()
    assert widget is not None
        