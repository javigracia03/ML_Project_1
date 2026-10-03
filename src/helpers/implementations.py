import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    N = tx.shape[0]
    for iter in range(max_iters):
        e = y - tx@w
        grad = - (1/N) * tx.T @ e
        w = w - gamma*grad

    final_e = y - tx@w
    mse = 1/2 * np.mean(final_e**2)
    return w, mse

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    w = initial_w
    N = tx.shape[0]
    for iter in range(max_iters):
        choice = int(np.random.random() * N)

        tx_sampled = tx[choice]
        y_sampled = y[choice]

        e = y_sampled - tx_sampled@w
        gradient = - tx_sampled*e
        w = w - gamma*gradient

    total_e = y - tx@w
    mse = (1/2) * np.mean(total_e**2)

    return w, mse
    

def least_squares(y, tx):
    a = tx.T @ tx
    b= tx.T @ y

    w = np.linalg.solve(a, b)

    e = y - tx @ w
    mse = (1/2) * np.mean(e**2)
    return w, mse


def ridge_regression(y, tx, lambda_):
    D = tx.shape[1]
    N = tx.shape[0]
    lambda_p = lambda_ * 2 * N

    a = lambda_p * np.eye(D) + tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)

    e = y - tx @ w
    mse = (1/2) * np.mean(e**2)

    return w, mse

