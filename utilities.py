import matplotlib.pyplot as plt


# Plot training history
def plot_metric(history_name, metric_name='accuracy', axis_label=None, graph_title=None, file_name="", dpi=100,
                xaxis_tick_label=None):
    """
    Function for plotting Neural network metrics, e.g. accuracy curve, loss curve, etc.
    :param history_name: Pointer for model history
    :param metric_name: accuracy, loss
    :param axis_label: [xaxis_label, yaxis_label]
    :param graph_title: Plot title
    :param file_name: Filename for saving file
    :param dpi: figure resolution
    :param xaxis_tick_label: Arbitrary xticks (in case of special names)
    :return: Nothing
    """
    metric = history_name.history[metric_name]
    validation_metric = history_name.history['val_' + metric_name]
    epochs = range(1, len(metric) + 1)
    plt.figure(figsize=plt.figaspect(1.), dpi=dpi)
    plt.plot(epochs, metric, 'bo', label='Training ' + metric_name.capitalize())
    plt.plot(epochs, validation_metric, 'r', label='Validation ' + metric_name.capitalize())
    if axis_label is None:
        axis_label = ['Epochs', 'met']
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    if graph_title is None:
        graph_title = metric_name.capitalize()
    plt.title(graph_title)
    if xaxis_tick_label:
        plt.xticks(epochs, xaxis_tick_label, rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    return

