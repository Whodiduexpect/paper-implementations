import autograd.numpy as np
from autograd import grad

INFO_EVERY_N_ITERS = 200
SHOW_GRADIENT = True
SHOW_PARAMS = True


def relu(x):
    return np.max(np.array([0, x]))


def feed_forward_layer(input_layer, hidden_layer, bias):
    return relu(np.sum((input_layer * hidden_layer)) + bias)


def adam(objective, stepsize, beta_1, beta_2, epsilon, params, n_iter):
    n_params = len(np.concatenate([x.ravel() for x in params]))

    m = np.zeros(n_params)
    v = np.zeros(n_params)
    gradient_fn = grad(objective)

    for t in range(n_iter):
        gradient_unflattened = gradient_fn(params)
        gradient = np.concatenate([x.ravel() for x in gradient_unflattened])

        m = beta_1 * m + (1 - beta_1) * gradient
        v = beta_2 * v + (1 - beta_2) * gradient**2

        m_hat = m / (1 - beta_1 ** (t + 1))
        v_hat = v / (1 - beta_2 ** (t + 1))

        # Okay, this flattening and unflattening is really a dumb hack
        params_flattened = np.concatenate([x.ravel() for x in params])

        params_flattened = params_flattened - stepsize * m_hat / (
            np.sqrt(v_hat) + epsilon
        )

        sizes = [np.prod(layer.shape) for layer in params]
        split_indices = np.cumsum(sizes)[:-1]
        flat_chunks = np.split(params_flattened, split_indices)
        params = [
            chunk.reshape(shape)
            for chunk, shape in zip(flat_chunks, [layer.shape for layer in params])
        ]

        loss = objective(params)

        if t % INFO_EVERY_N_ITERS == 0:
            print(f"Iteration {t}")
            if SHOW_GRADIENT:
                print(f"Gradient:\n{gradient_unflattened}\n")
            if SHOW_PARAMS:
                print(f"Params:\n{params}\n")
            print(f"Loss: {loss}\n\n")
    return params


def softmax(logits):
    logits = logits - np.max(logits)
    return np.exp(logits) / np.sum(np.exp(logits))


def log_softmax(logits):
    logits_max = np.max(logits)
    return logits - logits_max - np.log(np.sum(np.exp(logits - logits_max)))


# Cross entropy loss from logits is needed for numerical stability
def cross_entropy_loss(logits, true_class):
    return -log_softmax(logits)[true_class]


# Not actually used in training
def feed_forward_inference(input, params):
    return softmax(feed_forward_logits(input, params))


def feed_forward_logits(input, params):
    layer_activations = input
    for layer_index in range(len(params)):
        layer_activations = np.array(
            [
                feed_forward_layer(
                    layer_activations,
                    params[layer_index][i][1:],
                    params[layer_index][i][0],
                )
                for i in range(params[layer_index].shape[0])
            ]
        )

    return np.array(layer_activations)


def train_loss(input, true_class, params):
    prediction = feed_forward_logits(input, params)

    return cross_entropy_loss(prediction, true_class)


def objective(params):
    loss = 0.0

    # XOR training example
    loss += train_loss(np.array([0.0, 0.0]), 0, params)
    loss += train_loss(np.array([0.0, 1.0]), 1, params)
    loss += train_loss(np.array([1.0, 0.0]), 1, params)
    loss += train_loss(np.array([1.0, 1.0]), 0, params)
    return loss


initial_params = [
    # The second dimesion should be the dimensionality of the previous layer
    # (plus one because it includes the bias)
    np.random.rand(2, 3),
    np.random.rand(1, 3),
    np.random.rand(2, 2),
]

print(f"Initial Params:\n{initial_params}\n")

params = adam(
    objective,
    0.01,
    0.9,
    0.98,
    10**-9,
    initial_params,
    15_000,
)
