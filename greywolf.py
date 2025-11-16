import numpy as np

# ----------------------------------------------
# Grey Wolf Optimizer (GWO)
# ----------------------------------------------

def gwo(obj_func, lb, ub, dim, n_wolves=20, max_iter=100):
    # Initialize wolf positions randomly
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))

    # Initialize alpha, beta, delta wolves
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")

    beta_pos = np.zeros(dim)
    beta_score = float("inf")

    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    # Convergence tracker
    convergence_curve = []

    # Main loop
    for t in range(max_iter):
        for i in range(n_wolves):

            # Ensure within bounds
            wolves[i] = np.clip(wolves[i], lb, ub)

            # Fitness evaluation
            fitness = obj_func(wolves[i])

            # Update alpha, beta, delta
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, wolves[i].copy()

            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, wolves[i].copy()

            elif fitness < delta_score:
                delta_score, delta_pos = fitness, wolves[i].copy()

        # Parameter 'a' decreases linearly
        a = 2 - t * (2 / max_iter)

        # Update positions of wolves
        for i in range(n_wolves):
            for j in range(dim):

                # Alpha
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha

                # Beta
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i][j])
                X2 = beta_pos[j] - A2 * D_beta

                # Delta
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i][j])
                X3 = delta_pos[j] - A3 * D_delta

                # New position = average of X1, X2, X3
                wolves[i][j] = (X1 + X2 + X3) / 3

        convergence_curve.append(alpha_score)

        # (Optional) print progress
        # print(f"Iteration {t+1}, Best Score = {alpha_score}")

    return alpha_pos, alpha_score, convergence_curve

# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

# Objective function (Sphere function)
def sphere(x):
    return np.sum(x**2)

# Problem settings
dim = 5
lb = -10
ub = 10

best_pos, best_score, curve = gwo(
    obj_func=sphere,
    lb=lb,
    ub=ub,
    dim=dim,
    n_wolves=25,
    max_iter=200
)

print("Best Position:", best_pos)
print("Best Score:", best_score)
