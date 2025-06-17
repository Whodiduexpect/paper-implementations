import math

import numpy as np


def relu(x):
    return max(0, x)


def feed_forward(input_layer, hidden_layer, bias):
    z = np.sum((input_layer * hidden_layer)) + bias
    return relu(z)


def adam(derivative, stepsize, beta_1, beta_2, epsilon, params, n_iter):
    m = np.zeros(4)
    v = np.zeros(4)
    for t in range(n_iter):
        gradient = derivative(params)
        m = beta_1 * m + (1 - beta_1) * gradient
        v = beta_2 * v + (1 - beta_2) * gradient**2

        m_hat = m / (1 - beta_1 ** (t + 1))
        v_hat = v / (1 - beta_2 ** (t + 1))

        params = params - stepsize * m_hat / (math.sqrt(v_hat) + epsilon)
    return params


def objective(params):
    input = np.array([0, 1, 0])
    # Expected output: 1

    prediction = feed_forward(input, params[1:], params[0])

    print(prediction)


objective([0, 1, 1, 0])
