import numpy as np

#TODO: add docstrings and documentation for each function

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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    N = tx.shape[0]

    for i in range(max_iters):
        gradient = (1 / N) * tx.T @ (_sigmoid(tx @ w) - y)
        w = w - gamma*gradient

    pred = tx @ w
    logistic_loss = -np.mean(y * np.log(_sigmoid(pred)) + (1 - y) * np.log(1 - _sigmoid(pred)))

    return w, logistic_loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    N = tx.shape[0]

    for i in range(max_iters):
        gradient = (1 / N) * tx.T @ (_sigmoid(tx @ w) - y)
        gradient += 2*lambda_ * w
        w = w - gamma*gradient

    pred = tx @ w
    logistic_loss = -np.mean(y * np.log(_sigmoid(pred)) + (1 - y) * np.log(1 - _sigmoid(pred)))
    logistic_loss += lambda_ * np.sum(w ** 2)
    return w, logistic_loss 

def _sigmoid(t):
    return 1 / (1 + np.exp(-t))

