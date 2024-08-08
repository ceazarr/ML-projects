import numpy as np
import matplotlib
matplotlib.use("tkAgg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.jit
from torchvision import datasets
from torchvision.transforms import ToTensor

def add_noise(image: np.array, gw=0.1):
    noise = np.random.normal(loc=0.0, scale=gw, size=image.shape)
    noisy_image = noise + image
    return np.clip(noisy_image, 0., 1.)

def add_noise_torch(image_batch, gw=0.1):
    noise = torch.randn_like(image_batch) * gw
    noisy_image = noise + image_batch
    return torch.clamp(noisy_image, 0., 1.)

class BaseAE(nn.Module):
    def __init__(self, base_size=28*28, latent=10):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(128*3*3, latent)
        )
        ### DECODER
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent, 128*3*3),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, images):
        return self.decode(self.encode(images))

    def encode(self, images):
        x = self.encoder_cnn(images)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

    def decode(self, latent):
        x = self.decoder_lin(latent)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    # load the data
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor())
    train_size = 50000
    train_data, val_data = torch.utils.data.random_split(
        train_data, (train_size, len(train_data) - train_size))

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1024, shuffle=False)
    tb, tl = next(iter(train_loader))
    image = add_noise(tb[0].squeeze().detach().numpy(), gw=0.3)
    
    # set up all tool
    network = BaseAE(base_size=tb[0].numel(), latent=10)
    optim = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # training loop
    epochs = 10
    tlosses = []
    vlosses = []
    for e in range(1, epochs+1):
        tloss_per_epoch = 0
        vloss_per_epoch = 0
        network.train()
        for feat, _ in train_loader:
            optim.zero_grad()
            noisy_feat = add_noise_torch(feat, gw=0.2)
            fw = network(noisy_feat)
            bloss = loss_fn(fw, feat)
            tloss_per_epoch += bloss.item()
            bloss.backward()
            optim.step()
        network.eval()
        with torch.no_grad():
            for feat, _ in val_loader:
                noisy_feat = add_noise_torch(feat, gw=0.2)
                fw = network(noisy_feat)
                vloss_per_epoch += loss_fn(fw, feat).item()

        tlosses.append(tloss_per_epoch / len(train_loader))
        vlosses.append(vloss_per_epoch / len(val_loader))

        print(f"EPOCH {e}/{epochs}: Train Loss: {tlosses[-1]:.6f} | Val Loss: {vlosses[-1]:.6f}")

    # now, predict classes:
    images_np = []
    images_noisy = []
    predictions = []
    network.eval()
    with torch.no_grad():
        for feat, _ in test_loader:
            noisy_feat = add_noise_torch(feat, gw=0.2)
            images_np.append(feat.numpy())
            images_noisy.append(noisy_feat.numpy())
            pred = network(noisy_feat)
            predictions.append(pred.numpy())

    images_np = np.squeeze(np.concatenate(images_np))
    images_noisy = np.squeeze(np.concatenate(images_noisy))
    predictions = np.squeeze(np.concatenate(predictions))
    
    #Plot your results in a (N,3)-grid plot; The first column should show the original image, the second column the image with added noise, the third should show the denoised image
    N = 5
    fig, axes = plt.subplots(N, 3, figsize=(10, 20))
    for i in range(N):
        axes[i, 0].imshow(images_np[i], cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(images_noisy[i], cmap='gray')
        axes[i, 1].set_title("Noisy")
        axes[i, 2].imshow(predictions[i], cmap='gray')
        axes[i, 2].set_title("Denoised")
    plt.tight_layout()
    plt.savefig("plot.pdf")
    plt.show()

    # save the model
    sm = torch.jit.script(network)
    sm.save("model.torch")
