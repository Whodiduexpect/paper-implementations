import autograd.numpy as np
from autograd import grad


def relu(x):
    return np.max(np.array([0, x]))


def feed_forward(input_layer, hidden_layer, bias):
    return relu(np.sum((input_layer * hidden_layer)) + bias)


def adam(objective, stepsize, beta_1, beta_2, epsilon, params, n_iter):
    m = np.zeros(params.shape)
    v = np.zeros(params.shape)
    gradient_fn = grad(objective)

    for t in range(n_iter):
        print(f"Iteration {t}")
        gradient = gradient_fn(params)

        print(f"Gradient:\n{gradient}\n")
        m = beta_1 * m + (1 - beta_1) * gradient
        v = beta_2 * v + (1 - beta_2) * gradient**2

        m_hat = m / (1 - beta_1 ** (t + 1))
        v_hat = v / (1 - beta_2 ** (t + 1))

        params = params - stepsize * m_hat / (np.sqrt(v_hat) + epsilon)

        print(f"Params:\n{params}\n")

        loss = objective(params)
        print(f"Loss: {loss}\n\n")
    return params


def cross_entropy_loss(expected, predicted):
    predicted_class = np.round(predicted)

    y = 1 if predicted_class == expected else 0
    p = np.abs((1 - predicted_class) - predicted)

    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, keepdims=True))
    return exp_logits / np.sum(exp_logits, keepdims=True)


def inference(input, params):
    activations = np.array(
        [
            feed_forward(input, params[i][1:], params[i][0])
            for i in range(params.shape[0])
        ]
    )

    return softmax(activations)


def train_loss(input, expected_activations, params):
    prediction = inference(input, params)

    loss = 0

    for i in range(len(prediction)):
        loss += cross_entropy_loss(expected_activations[i], prediction[i])

    return loss


def objective(params):
    loss = 0

    # 1 if pattern is symmetrical, 0 if not
    loss += train_loss(np.array([0, 1, 0]), np.array([1, 0]), params)
    loss += train_loss(np.array([1, 1, 0]), np.array([0, 1]), params)
    loss += train_loss(np.array([0, 1, 1]), np.array([0, 1]), params)
    loss += train_loss(np.array([1, 0, 1]), np.array([1, 0]), params)
    loss += train_loss(np.array([1, 0, 0]), np.array([0, 1]), params)
    loss += train_loss(np.array([0, 0, 1]), np.array([0, 1]), params)
    loss += train_loss(np.array([1, 1, 1]), np.array([1, 0]), params)
    loss += train_loss(np.array([0, 0, 0]), np.array([1, 0]), params)

    return loss


params = adam(
    objective,
    0.01,
    0.9,
    0.98,
    10**-9,
    np.array(
        [
            # Random values
            [0.2, 0.0910, 0.5715, 0.9359],
            [0.0897, 2.786, 0.4657, 0.6974],
        ]
    ),
    300,
)
