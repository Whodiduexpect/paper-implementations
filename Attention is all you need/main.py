import numpy as np


def relu(x):
    if x <= 0:
        return 0
    return x


def feed_forward(input_layer, hidden_layer, bias):
    return relu((np.sum((input_layer * hidden_layer)) + bias))
