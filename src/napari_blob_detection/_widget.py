from magicgui import magicgui, widgets
from ._detect import difference_of_gaussian, laplacian_of_gaussian


METHODS = {
    'Difference of Gaussian': difference_of_gaussian,
    'Laplacian of Gaussian': laplacian_of_gaussian,
}


def detect_blobs_widget() -> widgets.Container:
    # Make a widget function that will select from the methods.
    method = widgets.ComboBox(choices=tuple(METHODS), name='method')
    container = widgets.Container(widgets=[method], labels=False)

    # When the method changes, populate the container with the correct widget.
    @method.changed.connect
    def _add_subwidget(method_name: str):
        if len(container) > 1:
            container.pop(-1).native.close()
        subwidget = magicgui(METHODS[method_name])
        subwidget.margins = (0, 0, 0, 0)
        container.append(subwidget)

    # Set the default method and force the subwidget to update accordingly.
    method.value = 'Difference of Gaussian'
    _add_subwidget(method.value)

    return container
