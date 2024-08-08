import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error
from torchinfo import summary


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )


    def forward(self, x):
        return self.network(x)


def get_dataloader(x: np.ndarray, y: np.ndarray) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


def train_model(model: torch.nn.Module,
                dataloader: DataLoader,
                epochs: int = 100) -> np.ndarray:
    model.train(True)
    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    # initialize vector to store losses per epoch
    loss_vec = np.zeros(epochs)

    # implement training loop
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
        loss_vec[epoch] = loss.item()
        print(f"Epoch {epoch:3d}: loss {loss_vec[epoch]}")

    return loss_vec


if __name__ == "__main__":

    interval = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
    target = np.sin(interval)
    
    dataloader = get_dataloader(interval, target)

    model = TorchModel()
    summary(model)

    epochs_per_plot = 20

    fig = plt.figure(figsize=(15, 10))
    fig.set_layout_engine("tight")
    all_losses = []

    for i in range(1, 9):
        lc = train_model(model, dataloader, epochs=epochs_per_plot)
        all_losses.extend(lc)
        prediction = model(torch.tensor(interval, dtype=torch.float32)).detach().numpy()

        # Create subplot for current epoch
        ax = fig.add_subplot(3, 3, i)
        ax.plot(interval, target, label="True Function")
        ax.plot(interval, prediction, linestyle="dashed", label="Model Prediction")
        ax.legend()
        mae = mean_absolute_error(prediction, target)
        ax.set_title(f"Epoch {i * epochs_per_plot} - MAE: {mae:.4f}")

    # Create subplot for learning curve
    ax = fig.add_subplot(3, 3, 9)
    ax.set_title("Learning Curve")
    ax.plot(np.arange(len(all_losses)), all_losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # Save plot and model
    plt.savefig('sine_fit.png')
    torch.save(model.state_dict(), 'sine_fit.pt')

    # Show plot
    plt.show()
