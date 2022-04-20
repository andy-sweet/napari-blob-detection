from magicgui import magicgui
from magicgui.widgets import ComboBox, Container
from ._detect import difference_of_gaussian, laplacian_of_gaussian


_METHODS = {
    'Difference of Gaussian': difference_of_gaussian,
    'Laplacian of Gaussian': laplacian_of_gaussian,
}


def detect_blobs_widget() -> Container:
    # Make a widget function that will select from the methods.
    methods = tuple(_METHODS.keys())
    method_combo = ComboBox(choices=methods, name='method')
    container = Container(widgets=[method_combo], labels=False)

    # When the method changes, populate the container with the correct widget.
    @method_combo.changed.connect
    def _add_subwidget(method_name: str):
        if len(container) > 1:
            container.pop(-1).native.close()
        subwidget = magicgui(_METHODS[method_name])
        subwidget.margins = (0, 0, 0, 0)
        container.append(subwidget)

    # Set the default method and force the subwidget to update accordingly.
    method_combo.value = methods[0]
    _add_subwidget(method_combo.value)

    return container
