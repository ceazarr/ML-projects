import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib


def ols_fit(x: np.ndarray, y: np.ndarray, basis_functions: list[callable]) -> np.ndarray:
    coeffs = np.zeros(len(basis_functions))
    N = x.shape[0]
    M = len(basis_functions)
    
    X = np.zeros((N, M))
    
    for i, basis_function in enumerate(basis_functions):
        X[:, i] = basis_function(x)
    coeffs = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return coeffs


def apply_fit(x: np.ndarray, coeffs: np.ndarray, basis_functions: list[callable]) -> np.ndarray:
    y_fit = np.zeros_like(x)
    for i, basis_function in enumerate(basis_functions):
        y_fit += coeffs[i] * basis_function(x)
    
    return y_fit


def plot_fit(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, basis_functions: list[callable]) -> None:
    x_range = np.linspace(-12, 12, 1000)
    y_range = apply_fit(x_range, coeffs, basis_functions)

    fig = plt.figure()
    fig.set_layout_engine("tight")
    ax = fig.add_subplot(111)
    ax.scatter(x, y, label="data")
    ax.plot(x_range, y_range, linestyle=":", color="red", label="ols fit")
    ax.legend()

    coeff_strings = [f"a_{i} = {c:-6.3f}" for i, c in enumerate(coeffs)]
    ax.text(1.05, 0.5, "\n".join(coeff_strings),
            verticalalignment="center", transform=ax.transAxes)


def main():
    current_directory = pathlib.Path(__file__).parent

    poly_data = pd.read_csv(current_directory / "polynomial.csv")
    poly_functions = [lambda x: np.ones_like(x), lambda x: x, lambda x: x**2, lambda x: x**3]
    poly_coeffs = ols_fit(poly_data["x"].to_numpy(),
                          poly_data["y"].to_numpy(),
                          poly_functions)
    plot_fit(poly_data["x"], poly_data["y"],
             poly_coeffs, poly_functions)
    plt.savefig(current_directory / "polynomial.png")

    wiggly_data = pd.read_csv(current_directory / "wiggly.csv")
    wiggly_functions = [lambda x: x, lambda x: x**2, lambda x: np.sin(x)]
    wiggly_coeffs = ols_fit(wiggly_data["x"].to_numpy(),
                            wiggly_data["y"].to_numpy(),
                            wiggly_functions)
    plot_fit(wiggly_data["x"], wiggly_data["y"],
             wiggly_coeffs, wiggly_functions)
    plt.savefig(current_directory / "wiggly.png")

    noisy_data = pd.read_csv(current_directory / "noisy.csv")
    noisy_functions = [lambda x: np.ones_like(x), lambda x: x, lambda x: x**2, lambda x: x**3]
    noisy_coeffs = ols_fit(noisy_data["x"].to_numpy(),
                           noisy_data["y"].to_numpy(),
                           noisy_functions)
    plot_fit(noisy_data["x"], noisy_data["y"],
             noisy_coeffs, noisy_functions)
    plt.savefig(current_directory / "noisy.png")

    return {"polynomial": poly_coeffs,
            "wiggly": wiggly_coeffs,
            "noisy": noisy_coeffs}


if __name__ == "__main__":
    results = main()
    print(results)
    plt.show()
