from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, nn, random, value_and_grad, vmap
from jax.lax import scan
from jax.tree_util import tree_map
from matplotlib.colors import ListedColormap

INFO_EVERY_N_ITERS = 100_000


@jit
def relu(x):
    return jnp.maximum(0, x)


@jit
def feed_forward_layer(input_layer, weights, beta, gamma, layernorm_eps=0.001):
    preactivations = jnp.dot(weights, input_layer)

    mean = jnp.mean(preactivations)
    variance = jnp.var(preactivations)

    normalized = (preactivations - mean) / jnp.sqrt(variance + layernorm_eps)

    return relu(gamma * normalized + beta)


@jit
def transformer_learning_rate(d_model, step_num, warmup_steps=4000):
    arg1 = step_num**-0.5
    arg2 = step_num * warmup_steps**-1.5
    return d_model**-0.5 * jnp.minimum(arg1, arg2)


@partial(jit, static_argnums=(0, 1, 2, 5))
def adam(beta_1, beta_2, epsilon, params, key, n_iter):
    d_model = determine_d_model(params)
    value_and_gradient_fn = value_and_grad(objective)

    m0 = tree_map(jnp.zeros_like, params)
    v0 = tree_map(jnp.zeros_like, params)

    initial_state = (params, m0, v0, key, 1)

    def update_step(state, _):
        params, m, v, key, t = state

        key, subkey = random.split(key)

        loss, gradient = value_and_gradient_fn(params, subkey)

        m = tree_map(
            lambda m_leaf, g_leaf: beta_1 * m_leaf + (1 - beta_1) * g_leaf, m, gradient
        )
        v = tree_map(
            lambda v_leaf, g_leaf: beta_2 * v_leaf + (1 - beta_2) * g_leaf**2,
            v,
            gradient,
        )

        m_hat = tree_map(lambda m_leaf: m_leaf / (1 - beta_1**t), m)
        v_hat = tree_map(lambda v_leaf: v_leaf / (1 - beta_2**t), v)

        learning_rate = transformer_learning_rate(d_model, t)

        new_params = tree_map(
            lambda p, m_h, v_h: p - learning_rate * m_h / (jnp.sqrt(v_h) + epsilon),
            params,
            m_hat,
            v_hat,
        )

        next_state = (new_params, m, v, key, t + 1)
        carry_out = loss

        return next_state, carry_out

    final_state, losses = scan(update_step, initial_state, None, length=n_iter)

    final_params, _, _, _, _ = final_state
    return final_params, losses


@jit
def softmax(logits):
    logits = logits - jnp.max(logits)
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits))


@jit
def log_softmax(logits):
    logits_max = jnp.max(logits)
    return logits - logits_max - jnp.log(jnp.sum(jnp.exp(logits - logits_max)))


@jit
def cross_entropy_loss(logits, true_class):
    return -log_softmax(logits)[true_class]


@jit
def feed_forward_inference(input, params):
    return softmax(feed_forward_logits(input, params))


@jit
def feed_forward_logits(input, params):
    layer_activations = input

    for layer in params:
        layer_activations = feed_forward_layer(
            layer_activations,
            layer[:, 2:],  # weights
            layer[:, 0],  # beta
            layer[:, 1],  # gamma
        )
    return layer_activations


@jit
def train_loss(input, true_class, params):
    prediction = feed_forward_logits(input, params)
    return cross_entropy_loss(prediction, true_class)


def determine_d_model(params):
    return max(params[0].shape[1] - 2, params[-1].shape[0])


@jit
def get_quadrant(point):
    x, y = point
    x_is_pos = x > 0
    y_is_pos = y > 0
    return 2 * (1 - y_is_pos) + (jnp.bitwise_xor(y_is_pos, x_is_pos))


@jit
def objective(params, key):
    point = random.normal(key, shape=(2,))
    true_class = get_quadrant(point)
    loss = train_loss(point, true_class, params)
    return loss


def analyze_and_visualize_model(
    params, key, n_test_points=300_000, n_bins=10, grid_resolution=1000
):
    print("\nAnalysis")

    key, test_key = random.split(key)
    test_points = random.normal(test_key, (n_test_points, 2))
    jnp_test_points = jnp.asarray(test_points)

    vmapped_inference = vmap(feed_forward_inference, in_axes=(0, None))

    true_labels = vmap(get_quadrant)(jnp_test_points)
    probabilities = vmapped_inference(jnp_test_points, params)
    confidences = jnp.max(probabilities, axis=1)
    predicted_labels = jnp.argmax(probabilities, axis=1)
    correct_predictions = predicted_labels == true_labels
    accuracy = jnp.mean(correct_predictions)
    print(f"\nOverall Test Accuracy: {accuracy:.2%}")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])
    ece = 0.0
    bin_accuracies, bin_confidences, bin_counts = [], [], []
    for i in range(n_bins):
        in_bin = bin_indices == i
        count = jnp.sum(in_bin)
        bin_counts.append(count)
        if count > 0:
            b_accuracy = jnp.mean(correct_predictions[in_bin])
            b_confidence = jnp.mean(confidences[in_bin])
            ece += (count / n_test_points) * jnp.abs(b_accuracy - b_confidence)
            bin_accuracies.append(b_accuracy)
            bin_confidences.append(b_confidence)
        else:
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_accuracies.append(0)
            bin_confidences.append(bin_center)
    print(f"Expected Calibration Error: {ece:.2%}\n")

    # Grid
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    # Get model logits for the grid
    vmapped_logits = vmap(feed_forward_logits, in_axes=(0, None))
    grid_logits = vmapped_logits(grid_points, params)

    # Get true labels for the grid
    grid_true_labels = vmap(get_quadrant)(grid_points)

    # Vectorize the loss function to get loss for every grid point
    vmapped_loss = vmap(cross_entropy_loss, in_axes=(0, 0))
    grid_loss = vmapped_loss(grid_logits, grid_true_labels)

    # Get other grid data for plotting
    grid_conf = jnp.max(nn.softmax(grid_logits, axis=1), axis=1)
    grid_classes = jnp.argmax(grid_logits, axis=1)

    # Reshape all grid data for plotting
    Z_loss = grid_loss.reshape(xx.shape)
    Z_conf = grid_conf.reshape(xx.shape)
    Z_classes = grid_classes.reshape(xx.shape)

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 15))
    fig.suptitle("Analysis", fontsize=20)

    # Reliability Diagram
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax1.plot(bin_confidences, bin_accuracies, "s-", c="blue", label="Model Calibration")
    ax1.set_title("Reliability Diagram")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy", color="blue")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1_twin = ax1.twinx()
    ax1_twin.bar(bin_confidences, bin_counts, width=1 / n_bins, alpha=0.2, color="gray")
    ax1_twin.set_ylabel("Number of Predictions", color="gray")
    ax1.legend(loc="upper left")

    # Decision Boundary
    ax2 = axes[0, 1]
    cmap_light = ListedColormap(["#FFBBBB", "#BBFFBB", "#BBBBFF", "#FFFFBB"])
    ax2.contourf(xx, yy, Z_classes, cmap=cmap_light, alpha=0.8)
    correct_mask = np.array(correct_predictions)
    ax2.scatter(
        test_points[correct_mask, 0],
        test_points[correct_mask, 1],
        s=10,
        c="black",
        alpha=0.2,
        label="Correct",
    )
    ax2.scatter(
        test_points[~correct_mask, 0],
        test_points[~correct_mask, 1],
        s=50,
        c="red",
        marker="x",
        label="Incorrect",
    )
    ax2.set_title("Decision Boundary with Misclassifications")
    ax2.set_xlabel("Input Feature 1 (x)")
    ax2.set_ylabel("Input Feature 2 (y)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Confidence Map
    ax3 = axes[1, 0]
    contour3 = ax3.contourf(xx, yy, Z_conf, cmap="viridis", levels=20)
    fig.colorbar(contour3, ax=ax3, label="Prediction Confidence")
    ax3.set_title("Confidence Map")
    ax3.set_xlabel("Input Feature 1 (x)")
    ax3.set_ylabel("Input Feature 2 (y)")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Loss Map
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(xx, yy, Z_loss, cmap="inferno", levels=20)
    fig.colorbar(contour4, ax=ax4, label="Cross-Entropy Loss")
    ax4.set_title("Loss Map")
    ax4.set_xlabel("Input Feature 1 (x)")
    ax4.set_ylabel("Input Feature 2 (y)")
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    key = random.key(1747190420)

    key, p1_key, p2_key = random.split(key, 3)
    initial_params = [
        random.uniform(p1_key, (16, 4)),
        random.uniform(p2_key, (4, 18)),
    ]

    print("Param Shape:", [p.shape for p in initial_params])

    n_iterations = 1_000_000
    key, adam_key = random.split(key)
    params, losses = adam(
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
        params=initial_params,
        key=adam_key,
        n_iter=n_iterations,
    )

    print("\n Training Progress Summary")
    for i in range(0, len(losses), INFO_EVERY_N_ITERS):
        if i == 0:
            continue
        it_num = i
        loss_val = losses[it_num - 1]
        print(f"Iteration {it_num}/{n_iterations}, Loss: {loss_val:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")

    key, analysis_key = random.split(key)
    analyze_and_visualize_model(params, analysis_key)
