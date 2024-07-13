import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def breit_wigner(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a / ((b - x)**2 + c)


def eval_jacobian(x: np.ndarray, beta: np.ndarray, function: callable, delta: float = 1e-4) -> np.ndarray:
    m, n = len(x), len(beta)
    if n != 3:
        raise ValueError("Input parameter 'beta' should have 3 elements (a, b, c) for the Breit-Wigner function.")
    # Initialize Jacobian matrix
    J = np.zeros((m, n))

    for j in range(n):
        # Create copies of beta with a small delta added and subtracted to the current parameter
        beta_delta_plus = beta.copy()
        beta_delta_plus[j] += delta
        beta_delta_minus = beta.copy()
        beta_delta_minus[j] -= delta

        # Calculate function values with the modified parameters
        f_delta_plus = function(x, *beta_delta_plus)
        f_delta_minus = function(x, *beta_delta_minus)

        # Calculate partial derivatives using central difference
        J[:, j] = (f_delta_plus - f_delta_minus) / (2 * delta)

    return J

def eval_error(x: np.ndarray, y: np.ndarray, beta: np.ndarray, function: callable) -> np.ndarray:
    f_x = function(x, *beta)
    error = y - f_x
    return error


def gauss_newton_update(jacobian: np.ndarray, error: np.ndarray) -> np.ndarray:
    # Solve for the parameter update using pseudo-inverse of Jacobian
    delta_beta = np.linalg.pinv(jacobian) @ error
    return delta_beta


def gauss_newton(x: np.ndarray,
                 y: np.ndarray,
                 beta: np.ndarray,
                 function: callable,
                 max_iter: int = 100,
                 threshold: float = 1e-3) -> tuple[np.ndarray, int]:
    for i in range(1, max_iter + 1):
        jac = eval_jacobian(x, beta, function)
        err = eval_error(x, y, beta, function)
        delta_beta = gauss_newton_update(jac, err)
        beta += delta_beta
        if np.linalg.norm(delta_beta) < threshold:   
            break
    if i == max_iter:
        raise UserWarning(f"Gauss-Newton method did not converge in {i} iterations.")
    return beta, i


def lm_update(jacobian: np.ndarray, error: np.ndarray, lm_lambda: float) -> np.ndarray:
    # Solve for the parameter update using damped least squares
    delta_beta = np.linalg.inv(jacobian.T @ jacobian + lm_lambda * np.eye(jacobian.shape[1])) @ jacobian.T @ error
    return delta_beta



def levenberg_marquardt(x: np.ndarray, y: np.ndarray, beta: np.ndarray, function: callable, max_iter: int = 1000, threshold: float = 1e-3) -> tuple[np.ndarray, int]:
    lm_lambda = 0.01  # Initial damping factor
    for i in range(1, max_iter + 1):
        jac = eval_jacobian(x, beta, function)
        err = eval_error(x, y, beta, function)
        
        # Solve the linear system for the parameter update
        A = jac.T @ jac + lm_lambda * np.eye(jac.shape[1])
        g = jac.T @ err
        delta_beta = np.linalg.solve(A, g)
        
        beta_new = beta + delta_beta
        err_new = eval_error(x, y, beta_new, function)
        
        if np.linalg.norm(err_new) < np.linalg.norm(err):
            beta = beta_new
            lm_lambda /= 8 
        else:
            lm_lambda *= 14  
        if np.linalg.norm(delta_beta) < threshold:
            break
    if i == max_iter:
        raise UserWarning(f"Levenberg-Marquardt method did not converge in {i} iterations.")
    return beta, i


if __name__ == "__main__":
    data = pd.read_csv(r"breit_wigner.csv")
    beta_guess_list = [
        [100000, 100, 1000],
        [80000, 100, 700],
        [50000, 100, 700],
        [80000, 150, 1000],
        [80000, 70, 700],
        [10000, 50, 500],
        [1000, 10, 100],
        [1, 1, 1]
    ]

    print("Testing Gauss-Newton method:")
    for beta_guess in beta_guess_list:
        print("initial guess", beta_guess)
        try:
            beta, iterations = gauss_newton(data["x"],
                                            data["y"],
                                            beta_guess,
                                            breit_wigner,
                                            max_iter=500)
            print(f"-> converged in {iterations:3d} iterations")
            print("->", beta)
        except Exception:
            print("-> did not converge")

    print("")
    print("Testing Levenberg-Marquardt method:")
    for beta_guess in beta_guess_list:
        print("initial guess", beta_guess)
        try:
            beta, iterations = levenberg_marquardt(data["x"],
                                                   data["y"],
                                                   beta_guess,
                                                   breit_wigner)
            print(f"-> converged in {iterations:3d} iterations")
            print("->", beta)
        except UserWarning:
            print("-> did not converge")
    
    x_range = np.linspace(-50, 250, 1000)
    y_range = breit_wigner(x_range, *beta)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data["x"], data["y"], label="data")
    
    ax.plot(x_range, y_range, ":r", label="fit")
    ax.legend()
    plt.show()
