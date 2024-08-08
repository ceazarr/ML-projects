import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 3),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 3)
        )

    def forward(self, x):
        return self.network(x)


def get_dataset(filename: str, feature_columns: list, label_columns: list) -> tuple:
    # Load dataset from the specified filename
    iris = pd.read_csv(r"C:\Users\ceaz\OneDrive\Desktop\Python Uni\pyeda24s_ex18-ge32peb\iris.csv")

    # Split data into training and test data
    train_data, test_data = train_test_split(iris, test_size=0.2, random_state=42)

    # Extract features and labels
    X_train = train_data[feature_columns].values
    Y_train = train_data[label_columns].values
    X_test = test_data[feature_columns].values
    Y_test = test_data[label_columns].values

    # Define scaler for features and encoder for labels
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False)

    # Fit scaler and encoder on training data
    X_train = scaler.fit_transform(X_train)
    Y_train = encoder.fit_transform(Y_train.reshape(-1, 1))

    # Transform both training and test data
    X_test = scaler.transform(X_test)
    Y_test = encoder.transform(Y_test.reshape(-1, 1))

    # Create TensorDataset objects for train and test sets
    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))

    return train, test


def train_model(model: torch.nn.Module, dataloader: DataLoader, epochs: int = 100):
    model.train(True)
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # initialize vector to store losses per epoch
    loss_vec = np.zeros(epochs)

    # implement training loop
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_vec[epoch] = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:3d}: loss {loss_vec[epoch]:.4f}")

    return loss_vec


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(y_batch, dim=1).cpu().numpy())
    return accuracy_score(all_labels, all_preds)


if __name__ == "__main__":
    # Define columns
    feature_columns = ['sl/cm', 'sw/cm', 'pl/cm', 'pw/cm']
    label_columns = ['class']

    # Get dataset
    train, test = get_dataset('iris.csv', feature_columns, label_columns)

    # Create dataloaders
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=False)

    # Train Model1
    model1 = Model1()
    loss_vec1 = train_model(model1, train_loader, epochs=300)
    torch.save(model1.state_dict(), 'model1.pt')

    # Train Model2
    model2 = Model2()
    loss_vec2 = train_model(model2, train_loader, epochs=300)
    torch.save(model2.state_dict(), 'model2.pt')

    # Evaluate models
    train_acc1 = evaluate_model(model1, train_loader)
    test_acc1 = evaluate_model(model1, test_loader)
    train_acc2 = evaluate_model(model2, train_loader)
    test_acc2 = evaluate_model(model2, test_loader)

    # Print accuracies
    print(f"Model1: Train Accuracy = {train_acc1:.4f}, Test Accuracy = {test_acc1:.4f}")
    print(f"Model2: Train Accuracy = {train_acc2:.4f}, Test Accuracy = {test_acc2:.4f}")

    # Scatter plots for Model1 and Model2 side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Model1 scatter plot
    train_data = train.tensors[0].numpy()
    train_labels = train.tensors[1].numpy()
    test_data = test.tensors[0].numpy()
    test_labels = test.tensors[1].numpy()

    ax1.scatter(train_data[:, 0], train_data[:, 1], c=np.argmax(train_labels, axis=1), cmap='viridis', label='Training Data')
    ax1.scatter(test_data[:, 0], test_data[:, 1], c=np.argmax(test_labels, axis=1), cmap='viridis', marker='x', label='Test Data')
    ax1.set_xlabel('Sepal Length (cm)')
    ax1.set_ylabel('Sepal Width (cm)')
    ax1.set_title(f'Model1 Scatter Plot\n'
                  f'Model1 Acc: Train={train_acc1:.2f}, Test={test_acc1:.2f}')
    ax1.legend()

    # Model2 scatter plot
    ax2.scatter(train_data[:, 0], train_data[:, 1], c=np.argmax(train_labels, axis=1), cmap='viridis', label='Training Data')
    ax2.scatter(test_data[:, 0], test_data[:, 1], c=np.argmax(test_labels, axis=1), cmap='viridis', marker='x', label='Test Data')
    ax2.set_xlabel('Sepal Length (cm)')
    ax2.set_ylabel('Sepal Width (cm)')
    ax2.set_title(f'Model2 Scatter Plot\n'
                  f'Model2 Acc: Train={train_acc2:.2f}, Test={test_acc2:.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('iris.png')
    plt.show()