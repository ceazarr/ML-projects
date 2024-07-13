import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def mandelbrot(c: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    if not isinstance(c, np.ndarray) or c.ndim != 2:
        raise ValueError("Input 'c' must be a 2D numpy array of complex numbers.")
    
    if np.any(np.abs(c) > 4):
        raise ValueError("The absolute value of complex numbers in 'c' must not exceed 4.")
    
    iterations = np.zeros_like(c, dtype=int)
    z = np.zeros_like(c, dtype=complex)
    
    for i in range(max_iter):
        mask = np.abs(z) <= 4
        z[mask] = z[mask] ** 2 + c[mask]
        iterations += mask.astype(int)
    
    return iterations


def plot_mandelbrot(c: np.ndarray, div_iter: np.ndarray) -> mpl.figure.Figure:
    fig = plt.figure()
    ax = fig.add_subplot()
    cplot = ax.pcolormesh(c.real, c.imag, div_iter, cmap="twilight")
    
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Mandelbrot Set")
    
    x_ticks = np.linspace(np.min(c.real), np.max(c.real), 5)
    y_ticks = np.linspace(np.min(c.imag), np.max(c.imag), 5)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_aspect("equal")
    
    cbar = fig.colorbar(cplot)
    cbar.set_label("Iterations Count")
    return fig

def get_complex_plane(center: complex, diameter: float, resolution: int = 800) -> np.ndarray:
    half_diameter = diameter / 2
    xmin = center.real - half_diameter
    xmax = center.real + half_diameter
    ymin = center.imag - half_diameter
    ymax = center.imag + half_diameter

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)

    X, Y = np.meshgrid(x, y)

    plane = X + 1j * Y

    return plane

if __name__ == "__main__":
    
    center1 = -0.5 + 0j
    diameter1 = 3
    resolution1 = 800
    plane1 = get_complex_plane(center1, diameter1, resolution1)
    mandel_image1 = mandelbrot(plane1, 250)
    mandel_figure1 = plot_mandelbrot(plane1, mandel_image1)
    plt.savefig("mandelbrot1.png")
    
    center2 = -1.4002 + 0j
    diameter2 = 0.005
    resolution2 = 800
    plane2 = get_complex_plane(center2, diameter2, resolution2)
    mandel_image2 = mandelbrot(plane2, 250)
    mandel_figure2 = plot_mandelbrot(plane2, mandel_image2)
    plt.savefig("mandelbrot2.png")
    
    plt.show() 
