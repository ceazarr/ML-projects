import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_contour_plot():
    # read and transform your data here
    data = pd.read_csv(r"fes.csv")
    #data = pd.read_csv("fes.csv")
    X = data["CV1"]
    Y = data["CV2"]
    Z = data["free energy (kJ/mol)"]


    # create your contour plot here
    fig = plt.figure()
    X_unique = np.unique(X)
    Y_unique = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
    Z_grid = Z.values.reshape(len(Y_unique), len(X_unique))
    fig, ax = plt.subplots()
    contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='plasma')

    cbar = plt.colorbar(contour, ax=ax, label='Free energy1000, J/mol)') 
    
    ax.set_xlabel('CV1')
    ax.set_ylabel('CV2')
    ax.set_title('Free energy (kJ/mol)')
    # save your plot
    plt.savefig("contour.png")


def create_surface_plot():
    # read and transform your data here
    data = pd.read_csv(r"C:\Users\ceaz\OneDrive\Desktop\Python Uni\pyeda24s_ex9-ge32peb\fes.csv")
    #data = pd.read_csv("fes.csv")
    X = data["CV1"]
    Y = data["CV2"]
    Z = data["free energy (kJ/mol)"]
    
    X_unique = np.unique(X)
    Y_unique = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)
    Z_grid = Z.values.reshape(len(Y_unique), len(X_unique))

    # create your 3D surface plot here
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='plasma')
    cbar = fig.colorbar(surf, ax=ax, label='Free Energy (kJ/mol)')
    
    # Set labels for axes
    ax.set_xlabel('CV1')
    ax.set_ylabel('CV2')
    ax.set_zlabel('Free Energy (kJ/mol)')
    # save your plot
    
    plt.savefig("surface.png")


if __name__ == "__main__":
    create_contour_plot()
    create_surface_plot()
